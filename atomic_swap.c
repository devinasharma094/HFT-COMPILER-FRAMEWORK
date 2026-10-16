#include <stdio.h>
#include <stdatomic.h>

typedef double (*model_fn)(double, double, double);

double baseline_model(double a, double b, double c)
{
    return a + b + c;
}

double specialized_model(double a, double b, double c)
{
    return (a + b + c);
}

/* Active model pointer */
_Atomic(model_fn) active_model = baseline_model;

void controlled_swap(model_fn new_model)
{
    atomic_store_explicit(
        &active_model,
        new_model,
        memory_order_release
    );
}

double execute_model(double a, double b, double c)
{
    model_fn model =
        atomic_load_explicit(
            &active_model,
            memory_order_acquire
        );

    return model(a, b, c);
}

int main(void)
{
    printf("Before swap: %.2f\n",
           execute_model(1.0, 2.0, 3.0));

    controlled_swap(specialized_model);

    printf("After swap: %.2f\n",
           execute_model(1.0, 2.0, 3.0));

    return 0;
}
