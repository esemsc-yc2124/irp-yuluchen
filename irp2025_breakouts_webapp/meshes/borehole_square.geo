// borehole_square.geo
// Square domain with central circular hole

// 参数设置
L = 0.1;       // 正方形边长 10 cm
a = 0.011;     // 孔半径（22 mm 直径）
res = 0.002;   // 网格大小（越小越精细）

// 定义圆中心和外围边
Point(1) = {0, 0, 0, res};
Point(2) = { a, 0, 0, res};
Point(3) = {0, a, 0, res};
Point(4) = {-a, 0, 0, res};
Point(5) = {0, -a, 0, res};

// 定义外方框四个点
Point(6) = {-L/2, -L/2, 0, res};
Point(7) = { L/2, -L/2, 0, res};
Point(8) = { L/2,  L/2, 0, res};
Point(9) = {-L/2,  L/2, 0, res};

// 圆边（4段）
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

// 外框（4条边）
Line(5) = {6, 7};  // bottom
Line(6) = {7, 8};  // right
Line(7) = {8, 9};  // top
Line(8) = {9, 6};  // left

// 建立曲面环
Line Loop(10) = {1, 2, 3, 4};        // 内圈：孔
Line Loop(11) = {5, 6, 7, 8};        // 外框
Plane Surface(12) = {11, 10};        // 用外减内建面

// 设置物理组（显式编号）
Physical Surface(1) = {12};              // FluidDomain
Physical Curve(15) = {1, 2, 3, 4};       // Circle (hole)
Physical Curve(11) = {5};                // bottom
Physical Curve(12) = {6};                // right
Physical Curve(13) = {7};                // top
Physical Curve(14) = {8};                // left