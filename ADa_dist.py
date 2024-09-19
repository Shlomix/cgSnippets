def adacons_hook(state, bucket: dist.GradBucket, sum_mode=False,
                 momentum_alpha=0.99, normalize_alpha_sum=True) -> torch.futures.Future[torch.Tensor]:
    
    g_i = bucket.buffer().clone()   # Current bucket gradients
    g_sum = bucket.buffer().clone() # Clone for async all-reduce

    # Asynchronous all-reduce operation on the current bucket's gradients
    g_sum_fut = dist.all_reduce(g_sum, async_op=True).get_future()

    def hook_callback(fut):
        # Take g_sum (all-reduced gradients)
        g_sum_val: torch.Tensor = fut.value()[0]

        # Update the global state with this bucket's contributions
        state['u_i'] += torch.sum(g_sum_val * g_i)
        state['norm_g_i'] += torch.sum(g_i ** 2)

        # Return the bucket's buffer (but wait for final symmetrical operation)
        return bucket.buffer()

    # Register the callback after the all-reduce operation
    bucket_future = g_sum_fut.then(hook_callback)

    # Track each bucket's future
    state['bucket_futures'].append(bucket_future)

    # Symmetrical operation after all buckets finish their contribution
    def symmetrical_operation_callback(_):
        # Compute the `alpha_i` based on the global state (u_i, norm_g_i)
        alpha_i = compute_alpha(state['u_i'], state['norm_g_i'])

        def apply_alpha_and_allreduce(fut):
            local_grads = bucket.buffer()  # Get the local gradients for the current bucket
            # Multiply the local gradients by `alpha_i`
            local_grads *= alpha_i

            # Perform an additional all-reduce on the scaled gradients
            return dist.all_reduce(local_grads, async_op=True).get_future().then(lambda _: local_grads)

        # Apply the `alpha_i` operation and perform all-reduce for each bucket
        all_buckets_futures = []
        for future in state['bucket_futures']:
            # Apply alpha to local gradients and all-reduce for each bucket's future
            bucket_modified_future = future.then(apply_alpha_and_allreduce)
            all_buckets_futures.append(bucket_modified_future)

        # Reset global state after the symmetrical operation is complete
        state['u_i'] = 0
        state['norm_g_i'] = 0

        # Return the modified bucket gradients
        return torch.futures.collect_all(all_buckets_futures)

    # Ensure that all buckets wait for the symmetrical operation to be completed
    all_buckets_done_future = torch.futures.collect_all(state['bucket_futures'])

    # Apply symmetrical operation after all bucket futures are done
    return all_buckets_done_future.then(symmetrical_operation_callback)
