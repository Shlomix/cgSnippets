import torch
import torch.distributed as dist
from torch.distributed import GradBucket
import torch.futures

class CustomState:
    """
    State to store the constant and iteration count for the hook.
    """
    def __init__(self, const):
        self.const = const
        self.iter = 0

def custom_comm_hook(state: CustomState, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Communication hook that computes the gradient transformation for each gradient tensor in the bucket.
    The operation performed is:
    g_bar - const * 2 / norm_squared_of(g_bar) * (1/num_workers * sum(norm_squared(g_i) * (g_i - g_bar)))
    """
    # Step 1: Extract gradients for all layers
    gradients = bucket.gradients()
    
    world_size = dist.get_world_size()
    futures = []

    def process_gradient(g_i):
        """
        Process a single gradient tensor following the specified transformation.
        """
        # Clone local gradient g_i
        g_bar = g_i.clone()

        # Step 2: Compute g_bar (average gradient across workers)
        fut = dist.all_reduce(g_bar, op=dist.ReduceOp.SUM, async_op=True).get_future()

        def compute_adjustment(fut):
            # g_bar is now the average gradient
            g_bar = fut.value() / world_size

            # Step 3: Compute norm_squared(g_bar)
            norm_squared_g_bar = torch.sum(g_bar ** 2)

            # Step 4: Compute norm_squared(g_i)
            norm_squared_g_i = torch.sum(g_i ** 2)

            # Step 5: Compute (norm_squared(g_i) * (g_i - g_bar))
            diff = g_i - g_bar
            adjustment_term = norm_squared_g_i * diff

            # Step 6: All-reduce adjustment term across workers
            return dist.all_reduce(adjustment_term, op=dist.ReduceOp.SUM, async_op=True).get_future()

        def finalize_adjustment(fut):
            # Get the final adjustment term after all-reduce
            adjustment_term = fut.value() / world_size

            # Compute final adjustment: const * 2 / norm_squared(g_bar) * adjustment_term
            adjustment = state.const * 2.0 * adjustment_term / torch.sum(g_bar ** 2)

            # Step 7: Compute new gradient as g_bar - adjustment
            new_grad = g_bar - adjustment

            return new_grad

        # Chain the futures for asynchronous execution
        return fut.then(compute_adjustment).then(finalize_adjustment)

    # Process all gradients in the bucket and collect their futures
    for grad in gradients:
        futures.append(process_gradient(grad))

    # Once all futures are done, return a combined future that updates all gradients
    def combine_results(futures):
        return torch.futures.collect_all(futures).then(lambda results: [r.value() for r in results])

    return combine_results(futures)

# Hook registration (assuming a model wrapped in DDP)
const_value = 0.5  # example constant, adjust as needed
state = CustomState(const_value)
model = torch.nn.parallel.DistributedDataParallel(your_model, device_ids=[your_device])
model.register_comm_hook(state, custom_comm_hook)
