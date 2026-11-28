"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
SOLUTION_CPP = r"""#include <bits/stdc++.h>
using namespace std;

// Shelf packing: every polyomino keeps its orientation (R = 0, F = 0) and is placed by its
// bounding box, tallest first, left to right on shelves of a fixed width.
int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<int> minx(n), miny(n), w(n), h(n);
    long long cells = 0;
    for (int i = 0; i < n; ++i) {
        int k;
        scanf("%d", &k);
        int x0 = INT_MAX, y0 = INT_MAX, x1 = INT_MIN, y1 = INT_MIN;
        for (int j = 0; j < k; ++j) {
            int x, y;
            scanf("%d %d", &x, &y);
            x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y);
        }
        minx[i] = x0; miny[i] = y0; w[i] = x1 - x0 + 1; h[i] = y1 - y0 + 1;
        cells += k;
    }
    long long width = max<long long>(*max_element(w.begin(), w.end()), (long long)ceil(sqrt(2.0 * cells)));
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return h[a] > h[b]; });
    vector<long long> X(n), Y(n);
    long long x = 0, y = 0, shelf = 0, used = 0;
    for (int i : order) {
        if (x + w[i] > width) { y += shelf; x = 0; shelf = 0; }
        X[i] = x - minx[i];
        Y[i] = y - miny[i];
        x += w[i];
        used = max(used, x);
        shelf = max<long long>(shelf, h[i]);
    }
    printf("%lld %lld\n", used, y + shelf);
    for (int i = 0; i < n; ++i) printf("%lld %lld 0 0\n", X[i], Y[i]);
    return 0;
}
"""
# EVOLVE-BLOCK-END
