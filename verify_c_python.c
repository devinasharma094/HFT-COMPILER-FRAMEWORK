#include <stdio.h>
#include <math.h>

double predict_baseline(double, double, double);

int main(void)
{
    double X[3][3] = {
        {0.001, 100.0, 50.0},
        {0.002, 150.0, 75.0},
        {0.003, 200.0, 100.0}
    };

    for (int i = 0; i < 3; i++)
    {
        double margin = predict_baseline(
            X[i][0], X[i][1], X[i][2]
        );

        double probability = margin;

        printf(
            "row %d: C margin = %.9f, C probability = %.9f\n",
            i, margin, probability
        );
    }

    return 0;
}
