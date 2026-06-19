#include <catch2/catch_test_macros.hpp>

#include <ai/core/batch.hpp>
#include <ai/model/generate_result.hpp>

#include <boost/asio.hpp>

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

namespace {

/// A fully offline BatchProcessor that drives status through a scripted
/// sequence, letting run_batch's submit/poll/results orchestration be tested
/// without any network.
class FakeBatchProcessor : public ai::batch::BatchProcessor {
public:
    explicit FakeBatchProcessor(int polls_until_complete)
        : polls_until_complete_(polls_until_complete) {}

    int submit_calls = 0;
    int status_calls = 0;
    int results_calls = 0;
    bool force_fail = false;

    ai::Task<std::string> submit(std::vector<ai::batch::BatchRequest>) override {
        ++submit_calls;
        co_return "batch-123";
    }

    ai::Task<ai::batch::BatchInfo> status(std::string_view id) override {
        (void)id;
        ++status_calls;
        ai::batch::BatchInfo info;
        info.id = "batch-123";
        if (force_fail) {
            info.status = ai::batch::BatchStatus::Failed;
        } else if (status_calls >= polls_until_complete_) {
            info.status = ai::batch::BatchStatus::Completed;
        } else {
            info.status = ai::batch::BatchStatus::InProgress;
        }
        co_return info;
    }

    ai::Task<std::vector<ai::batch::BatchResponseItem>> results(std::string_view id) override {
        (void)id;
        ++results_calls;
        co_return std::vector<ai::batch::BatchResponseItem>{
            ai::batch::BatchResponseItem{.custom_id = "r1"},
        };
    }

    ai::Task<void> cancel(std::string_view) override { co_return; }
    ai::Task<std::vector<ai::batch::BatchInfo>> list(int) override { co_return {}; }

private:
    int polls_until_complete_;
};

template <typename T>
T run(ai::Task<T> task, boost::asio::io_context& ioc) {
    task.start();
    int guard = 0;
    while (!task.done() && guard++ < 100'000) {
        ioc.run_one();
    }
    REQUIRE(task.done());
    return task.get();
}

ai::batch::BatchRunOptions make_options(
    ai::batch::BatchProcessorPtr p, boost::asio::io_context& ioc, int max_polls
) {
    return ai::batch::BatchRunOptions{
        .processor = std::move(p),
        .requests = {ai::batch::BatchRequest{.custom_id = "r1"}},
        .io_context = ioc,
        .poll_interval = std::chrono::milliseconds(0), // no real waiting
        .max_polls = max_polls,
    };
}

ai::Task<ai::batch::BatchRunResult> do_run(ai::batch::BatchRunOptions opts) {
    co_return co_await ai::batch::run_batch(std::move(opts));
}

} // namespace

TEST_CASE("run_batch submits, polls to completion, and fetches results", "[batch]") {
    boost::asio::io_context ioc;
    auto proc = std::make_shared<FakeBatchProcessor>(2);

    auto result = run(do_run(make_options(proc, ioc, 10)), ioc);

    REQUIRE(result.batch_id == "batch-123");
    REQUIRE(result.final_status.status == ai::batch::BatchStatus::Completed);
    REQUIRE(result.results.size() == 1);
    REQUIRE(proc->submit_calls == 1);
    REQUIRE(proc->status_calls == 2);
    REQUIRE(proc->results_calls == 1);
}

TEST_CASE("run_batch propagates failure without fetching results", "[batch]") {
    boost::asio::io_context ioc;
    auto proc = std::make_shared<FakeBatchProcessor>(1);
    proc->force_fail = true;

    auto result = run(do_run(make_options(proc, ioc, 10)), ioc);

    REQUIRE(result.final_status.status == ai::batch::BatchStatus::Failed);
    REQUIRE(result.results.empty());
    REQUIRE(proc->results_calls == 0);
}

TEST_CASE("run_batch throws when max_polls is exceeded", "[batch]") {
    boost::asio::io_context ioc;
    auto proc = std::make_shared<FakeBatchProcessor>(1'000'000); // never completes

    REQUIRE_THROWS_AS(run(do_run(make_options(proc, ioc, 3)), ioc), std::runtime_error);
    REQUIRE(proc->results_calls == 0);
}
