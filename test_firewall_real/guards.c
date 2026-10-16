/* Auto-generated interval guards from the VEC (exact doubles; identical to the box used for elimination) */
#include <stdbool.h>

static inline bool guard_calm(double realized_vol, double trade_intensity, double avg_trade_size) {
    return (realized_vol >= 2.964887488445281e-06 && realized_vol <= 3.0052423374418236e-05) && (trade_intensity >= 3.26 && trade_intensity <= 22.27) && (avg_trade_size >= 1677.6178167247276 && avg_trade_size <= 4685.2862534463175);
}

static inline bool guard_normal(double realized_vol, double trade_intensity, double avg_trade_size) {
    return (realized_vol >= 3.2500718817508836e-05 && realized_vol <= 5.371523950454927e-05) && (trade_intensity >= 15.703333333333333 && trade_intensity <= 44.36) && (avg_trade_size >= 1682.6495128056938 && avg_trade_size <= 4825.72626559685);
}

static inline bool guard_volatile(double realized_vol, double trade_intensity, double avg_trade_size) {
    return (realized_vol >= 5.6787675753796985e-05 && realized_vol <= 0.000173736649910096) && (trade_intensity >= 34.13999999999998 && trade_intensity <= 209.21) && (avg_trade_size >= 1792.0129207514176 && avg_trade_size <= 5224.976848935749);
}
