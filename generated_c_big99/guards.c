/* Auto-generated interval guards from the VEC (exact doubles; identical to the box used for elimination) */
#include <stdbool.h>

static inline bool guard_calm(double realized_vol, double trade_intensity, double avg_trade_size) {
    return (realized_vol >= 8.052356646082148e-07 && realized_vol <= 3.128073029436787e-05) && (trade_intensity >= 1.75 && trade_intensity <= 30.619666666666745) && (avg_trade_size >= 1035.6956301118569 && avg_trade_size <= 6679.101124881111);
}

static inline bool guard_normal(double realized_vol, double trade_intensity, double avg_trade_size) {
    return (realized_vol >= 3.151289787272815e-05 && realized_vol <= 5.501155908697589e-05) && (trade_intensity >= 11.54 && trade_intensity <= 62.419633333332996) && (avg_trade_size >= 1178.3189052213527 && avg_trade_size <= 6501.10591560609);
}

static inline bool guard_volatile(double realized_vol, double trade_intensity, double avg_trade_size) {
    return (realized_vol >= 5.533048006908229e-05 && realized_vol <= 0.0003314533766989318) && (trade_intensity >= 25.816666666666666 && trade_intensity <= 486.71833333332944) && (avg_trade_size >= 1366.4522699332674 && avg_trade_size <= 7140.701465749267);
}
