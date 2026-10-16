#include <math.h>

/* PROTECTED_RISK_REGION_BEGIN */

#define MAX_POSITION 100.0
#define MAX_ORDER_SIZE 10.0

int risk_check(double position, double order_size)
{
    if (fabs(position + order_size) > MAX_POSITION)
        return 0;

    if (fabs(order_size) > MAX_ORDER_SIZE)
        return 0;

    return 1;
}

/* PROTECTED_RISK_REGION_END */
