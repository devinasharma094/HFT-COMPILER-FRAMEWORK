#include <stdio.h>
#ifndef HFT_SILENT
#define HFT_SILENT 0
#endif

double predict_dispatch(double realized_vol,
                        double trade_intensity,
                        double avg_trade_size);

int risk_check(double position, double order_size);

typedef enum {
    ACTION_HOLD = 0,
    ACTION_BUY  = 1,
    ACTION_SELL = -1
} Action;

Action trading_policy(double probability)
{
    if (probability >= 0.55)
        return ACTION_BUY;

    if (probability <= 0.45)
        return ACTION_SELL;

    return ACTION_HOLD;
}

int execute_decision(double realized_vol,
                     double trade_intensity,
                     double avg_trade_size,
                     double position,
                     double order_size)
{
    double probability = predict_dispatch(
        realized_vol,
        trade_intensity,
        avg_trade_size
    );

    Action action = trading_policy(probability);

    if (action == ACTION_HOLD)
    {
        if (!HFT_SILENT) printf("ML probability: %.9f -> HOLD\n", probability);
        return 0;
    }

    double signed_order =
        (action == ACTION_BUY) ? order_size : -order_size;

    if (!risk_check(position, signed_order))
    {
        if (!HFT_SILENT) printf("ML probability: %.9f -> %s -> RISK REJECT\n",
               probability,
               action == ACTION_BUY ? "BUY" : "SELL");
        return 0;
    }

    if (!HFT_SILENT) printf("ML probability: %.9f -> %s -> ORDER ACCEPTED: %.2f\n",
           probability,
           action == ACTION_BUY ? "BUY" : "SELL",
           signed_order);

    return 1;
}
