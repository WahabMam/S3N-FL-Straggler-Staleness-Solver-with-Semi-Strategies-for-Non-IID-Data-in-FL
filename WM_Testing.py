# import threading

# def slow():
#     # stragglers
#     for i in range(3):
#         print(f'Slow function is running for {i} local rounds ')


# def fast():
#     # non-Stragglers
#     print('Fast function is running')


# threads = []
 
# for i in range(5): # global epochs
#     slow_thread = threading.Thread(target=slow)
#     fast_thread = threading.Thread(target=fast)
    
#     threads.append(fast_thread)
#     threads.append(slow_thread)

    
#     fast_thread.start()
#     slow_thread.start()
    

# for thread in threads:
#     thread.join()


# import threading
# import time

# def slow():
#     # Stragglers
#     for i in range(3):
#         print(f'Slow function is running for {i} local rounds')
#         time.sleep(1)  # Simulating slower execution

# def fast():
#     # Non-Stragglers
#     print('Fast function is running')
#     time.sleep(0.5)  # Simulating a quicker execution

# for i in range(5):  # Global epochs
#     print(f"--- Global Epoch {i} ---")

#     slow_thread = threading.Thread(target=slow)
#     fast_thread = threading.Thread(target=fast)

#     slow_thread.start()
#     fast_thread.start()

#     # Wait for both to complete before moving to the next global epoch
#     slow_thread.join()
#     fast_thread.join()


####################################################################################################
import threading
import time

# Function for fast clients (Non-stragglers)
# def fast_client(global_epoch):
#     print(f"[Fast Client] Global Epoch {global_epoch} started.")
#     time.sleep(2)  # Simulate training time
#     print(f"[Fast Client] Global Epoch {global_epoch} finished.")

# # Function for slow clients (Stragglers)
# def slow_client(global_epoch):
#     print(f"[Slow Client] Global Epoch {global_epoch} started.")
#     for local_epoch in range(3):  # Simulating 3 local epochs
#         print(f"   [Slow Client] Local Epoch {local_epoch} running...")
#         time.sleep(1)  # Simulating local training time
#     print(f"[Slow Client] Global Epoch {global_epoch} finished after local training.")

# # Simulate 5 global epochs
# for global_epoch in range(5):
#     print(f"\n--- Global Epoch {global_epoch} ---")

#     # Start both fast and slow clients **simultaneously**
#     fast_thread = threading.Thread(target=fast_client, args=(global_epoch,))
#     slow_thread = threading.Thread(target=slow_client, args=(global_epoch,))

#     fast_thread.start()
#     slow_thread.start()

#     # Wait for both to finish before moving to the next global epoch
#     fast_thread.join()
#     slow_thread.join()
##################################################################################################################

import threading
import time

# Shared global epoch counter
global_epoch = 0
MAX_GLOBAL_EPOCHS = 5
lock = threading.Lock()  # Ensures thread-safe updates

# Function for fast clients (Non-stragglers)
def fast_clients():
    global global_epoch

    while global_epoch < MAX_GLOBAL_EPOCHS:
        with lock:  # Ensure exclusive access to global_epoch
            current_epoch = global_epoch
            global_epoch += 1  # Increment the global epoch for next iteration

        print(f"\n--- Global Epoch {current_epoch} ---", flush=True)
        print(f"[Fast Client] Global Epoch {current_epoch} started.", flush=True)
        
        time.sleep(1)  # Simulate fast client training
        print(f"[Fast Client] Global Epoch {current_epoch} finished.", flush=True)

# Function for slow clients (Stragglers) - Runs Independently
def slow_clients():
    global global_epoch
    slow_epoch = 0

    while slow_epoch < MAX_GLOBAL_EPOCHS:
        with lock:
            if slow_epoch >= global_epoch:
                continue  # Wait until a new global epoch starts

        print(f"[Slow Client] Global Epoch {slow_epoch} started.", flush=True)

        for local_epoch in range(3):  # Simulating 3 local epochs
            print(f"   [Slow Client] Local Epoch {local_epoch} running...", flush=True)
            time.sleep(1)  # Simulating local training time

        print(f"[Slow Client] Global Epoch {slow_epoch} finished after local training.", flush=True)
        slow_epoch += 1  # Move to the next slow epoch

# Start fast and slow clients
fast_thread = threading.Thread(target=fast_clients)
slow_thread = threading.Thread(target=slow_clients)

fast_thread.start()
slow_thread.start()

# Wait for both threads to finish
fast_thread.join()
slow_thread.join()

print("Training completed.")

