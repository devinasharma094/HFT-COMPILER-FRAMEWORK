#include <stdio.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>

typedef int (*model_fn)(void);

int model_a(void)
{
    return 1;
}

int model_b(void)
{
    return 2;
}

_Atomic(model_fn) active_model = model_a;

void controlled_swap(model_fn new_model)
{
    atomic_store_explicit(
        &active_model,
        new_model,
        memory_order_release
    );
}

void *worker(void *arg)
{
    (void)arg;

    for (int i = 0; i < 1000000; i++)
    {
        model_fn model =
            atomic_load_explicit(
                &active_model,
                memory_order_acquire
            );

        int result = model();

        if (result != 1 && result != 2)
        {
            printf("INVALID RESULT: %d\n", result);
            return NULL;
        }
    }

    return NULL;
}

int main(void)
{
    pthread_t t1, t2;

    pthread_create(&t1, NULL, worker, NULL);
    pthread_create(&t2, NULL, worker, NULL);

    usleep(1000);

    controlled_swap(model_b);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Concurrent atomic swap test: PASS\n");

    return 0;
}
