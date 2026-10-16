/* Auto-generated dispatch: tries each regime guard in turn, falls back to baseline */
#include "guards.c"

double predict_baseline(double, double, double);
double predict_calm(double, double, double);
double predict_normal(double, double, double);
double predict_volatile(double, double, double);

double predict_dispatch(double realized_vol, double trade_intensity, double avg_trade_size) {
    if (guard_calm(realized_vol, trade_intensity, avg_trade_size)) {
        return predict_calm(realized_vol, trade_intensity, avg_trade_size);
    }
    if (guard_normal(realized_vol, trade_intensity, avg_trade_size)) {
        return predict_normal(realized_vol, trade_intensity, avg_trade_size);
    }
    if (guard_volatile(realized_vol, trade_intensity, avg_trade_size)) {
        return predict_volatile(realized_vol, trade_intensity, avg_trade_size);
    }
    return predict_baseline(realized_vol, trade_intensity, avg_trade_size);  /* fallback */
}
