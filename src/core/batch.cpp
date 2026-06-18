#include <ai/core/batch.hpp>

#include <boost/asio.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/system/error_code.hpp>

#include <coroutine>
#include <stdexcept>
#include <utility>

namespace ai::batch {

namespace {

// Awaits a steady_timer on the given io_context. Completion is delivered on the
// io_context thread, so it integrates with the standard start()/run_one() drive
// loop (unlike a detached sleep thread, which would not wake a pure-sync loop).
struct TimerAwaitable {
    boost::asio::io_context& ioc;
    std::chrono::milliseconds duration;

    bool await_ready() const noexcept { return duration.count() <= 0; }

    void await_suspend(std::coroutine_handle<> h) const {
        // Keep the timer alive until the completion handler fires.
        auto timer = std::make_shared<boost::asio::steady_timer>(ioc, duration);
        timer->async_wait([timer, h](boost::system::error_code) {
            h.resume();
        });
    }

    void await_resume() const noexcept {}
};

} // namespace

Task<BatchRunResult> run_batch(BatchRunOptions options) {
    if (!options.processor) {
        throw std::invalid_argument("run_batch: processor must not be null");
    }

    std::string batch_id =
        co_await options.processor->submit(std::move(options.requests));

    BatchInfo info{};
    int polls = 0;

    while (true) {
        options.cancel.throw_if_cancelled();

        info = co_await options.processor->status(batch_id);

        if (info.status == BatchStatus::Completed ||
            info.status == BatchStatus::Failed ||
            info.status == BatchStatus::Cancelled ||
            info.status == BatchStatus::Expired) {
            break;
        }

        if (options.max_polls && ++polls >= *options.max_polls) {
            throw std::runtime_error("run_batch: exceeded max_polls waiting for completion");
        }

        co_await TimerAwaitable{options.io_context, options.poll_interval};
    }

    std::vector<BatchResponseItem> results;
    if (info.status == BatchStatus::Completed) {
        results = co_await options.processor->results(batch_id);
    }

    co_return BatchRunResult{
        .batch_id = std::move(batch_id),
        .results = std::move(results),
        .final_status = std::move(info),
    };
}

} // namespace ai::batch
