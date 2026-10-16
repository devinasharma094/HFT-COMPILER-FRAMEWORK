#include <stdio.h>

int execute_decision(double realized_vol,
                     double trade_intensity,
                     double avg_trade_size,
                     double position,
                     double order_size);

int main(void)
{
    double realized_vol = 0.001;
    double trade_intensity = 100.0;
    double avg_trade_size = 50.0;

    double position = 95.0;
    double order_size = 10.0;

    printf("=== HFT ML DECISION ===\n");

    int result = execute_decision(
        realized_vol,
        trade_intensity,
        avg_trade_size,
        position,
        order_size
    );

    printf("Decision result: %d\n", result);

    return 0;
}
