#include <stdio.h>

int risk_check(double position, double order_size);

int main(void)
{
    printf("safe: %d\n", risk_check(50.0, 5.0));
    printf("position violation: %d\n", risk_check(95.0, 10.0));
    printf("order violation: %d\n", risk_check(50.0, 15.0));

    return 0;
}
