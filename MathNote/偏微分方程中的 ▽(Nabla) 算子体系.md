# 偏微分方程中的 ∇（Nabla）算子体系

> 个人学习笔记：把偏微分方程里反复出现的 **∇ 算子**整理成一套能直接拿来计算的语言。符号 **▽** 与 **∇** 是同一个算子，下文统一写成 ∇。
>
> 三种基本运算是梯度、散度、旋度。梯度回答“标量场往哪边变得最快”，散度回答“这里有没有源或汇”，旋度回答“这里在不在转”。拉普拉斯算子是散度再作用在梯度上的结果，也就是 ∇ 与自己做点乘。
>
> 全文默认三维直角坐标 $(x,y,z)$，向量场写成 $\mathbf{F} = (F_x, F_y, F_z)$，其中 $F_x,F_y,F_z$ 是 $\mathbf{F}$ 沿三条坐标轴的分量。$\partial_x$ 是 $\partial/\partial x$ 的简写，$\partial_{xx}$ 是 $\partial^2/\partial x^2$ 的简写，其他变量同样。物理文献里拉普拉斯算子取 $\nabla^2 = \partial_{xx}+\partial_{yy}+\partial_{zz}$；微分几何里有时把 $\Delta$ 定义成 $-\nabla^2$，读文献时先看符号约定。

## 目录

- 一、∇（Nabla / Del 算子）
- 二、梯度 $\nabla f$
- 三、散度 $\nabla \cdot \mathbf{F}$
- 四、旋度 $\nabla \times \mathbf{F}$
- 五、拉普拉斯算子 $\nabla^2$
- 六、实际算例
- 附录：记号、关系图与速记

三种运算分列在第二、三、四节，公式、几何意义和积分定理并列。第六节全部可以手算核对。附录保留原来的记号说明、关系图和速记。

---

## 一、∇（Nabla / Del 算子）

### 1.1 定义

∇ 是一个向量形状的微分符号，单独写出来还不是一次具体运算。它只有作用到标量场或向量场上，才得到梯度、散度或旋度。

直角坐标里把它写成

$$\nabla = \left(\frac{\partial}{\partial x},\ \frac{\partial}{\partial y},\ \frac{\partial}{\partial z}\right).$$

把它当成“带着偏导的形式向量”来参与乘法：

| 结合方式 | 写法 | 名称 | 输入 → 输出 |
|----------|------|------|-------------|
| 直接作用 | $\nabla f$ | 梯度 | 标量场 → 向量场 |
| 点乘 | $\nabla \cdot \mathbf{F}$ | 散度 | 向量场 → 标量场 |
| 叉乘 | $\nabla \times \mathbf{F}$ | 旋度 | 向量场 → 向量场 |

点乘、叉乘的代数规则照用，但每个分量里的 $\partial/\partial x_i$ 必须作用到它右边的函数上，不能和普通数字对调后把导数丢掉。下标 $i$ 只是编号，$x_i$ 依次代表 $x,y,z$。

### 1.2 和一元微积分的对应

| 一元运算 | 多元类比 | 积分定理 |
|----------|----------|----------|
| $f'(x)$ | $\nabla f$ | $\displaystyle\int_C \nabla f\cdot d\mathbf{r} = f(B)-f(A)$ |
| 看 $F$ 的增减 | $\nabla \cdot \mathbf{F}$（流出减流入） | 散度定理 |
| 看转弯 | $\nabla \times \mathbf{F}$（环量面密度） | 斯托克斯定理 |

表中 $C$ 是积分路径，$A$、$B$ 是路径两端，$d\mathbf{r}$ 是沿路径切向的位移微元（不是 $r$ 的微分记号单独使用）。散度定理和斯托克斯定理在第三、四节写出。它们是把“一点处的导数”还原成“一小块区域边界上的通量或环量”的依据，也是格林公式进入偏微分方程的入口。

---

## 二、梯度 $\nabla f$

### 2.1 定义与计算

设 $f$ 在一点附近偏导连续。梯度是

$$\nabla f = \left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y},\ \frac{\partial f}{\partial z}\right).$$

计算时对三个自变量各求一阶偏导，按原顺序排成向量。

沿单位向量 $\mathbf{u}$（长度为 $1$）的方向导数 $D_{\mathbf{u}} f$ 等于梯度在该方向上的投影：

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}.$$

因此：

- $\nabla f$ 指向 $f$ **上升最快**的方向；
- $|\nabla f|$ 是向量 $\nabla f$ 的长度，也就是这个最快上升率；
- 沿 $-\nabla f$ 下降最快；
- 与 $\nabla f$ 垂直的方向上，方向导数为 $0$，也就是沿着等值面走动时 $f$ 不变。

全微分把这条投影写成

$$df = \nabla f \cdot d\mathbf{r} = f_x\,dx + f_y\,dy + f_z\,dz.$$

下标 $f_x$ 就是 $\partial f/\partial x$，不是 $\mathbf{F}$ 那种向量分量；$f_y$、$f_z$ 同理。

### 2.2 等值面与力场

$f(x,y,z) = c$（$c$ 是常数）是一张等值面。面上任意切向量 $\mathbf{t}$ 都满足 $\nabla f \cdot \mathbf{t} = 0$，所以 **梯度垂直于等值面**，由低值一侧指向高值一侧。

若把 $f$ 看成势能，力常写成

$$\mathbf{F} = -\nabla f.$$

负号表示力指向势能下降的方向。静电场用的就是这个约定：$\mathbf{E}$ 是电场，$\varphi$ 是电位，$\mathbf{E} = -\nabla \varphi$。判断“某向量场是不是梯度”时，要先看旋度是否为零，见第四节。

### 2.3 梯度定理

$C$ 是从 $A$ 到 $B$ 的分段光滑曲线时，

$$\int_C \nabla f \cdot d\mathbf{r} = f(B) - f(A).$$

线积分只依赖两端点，与路径无关。反过来，若向量场在合适的区域上是某个标量的梯度，它的线积分也只依赖端点。

### 2.4 乘积法则

$$\nabla(fg) = f\nabla g + g\nabla f, \qquad \nabla\!\left(\frac{f}{g}\right) = \frac{g\nabla f - f\nabla g}{g^2}\ (g\neq 0).$$

链式法则：若 $f = h(u)$，而 $u$ 是空间点的标量函数，$h'$ 是 $h$ 对这个自变量的普通导数，则 $\nabla f = h'(u)\,\nabla u$。位置向量 $\mathbf{r}=(x,y,z)$ 的长度记为 $r=|\mathbf{r}|=\sqrt{x^2+y^2+z^2}$。此时 $\nabla r = \mathbf{r}/r$，从而

$$\nabla\!\left(\frac{1}{r}\right) = -\frac{\mathbf{r}}{r^3}\qquad (r\neq 0).$$

这个式子在第六节用来核对泊松方程的基本解。

---

## 三、散度 $\nabla \cdot \mathbf{F}$

### 3.1 定义与计算

散度是 ∇ 与向量场的点乘，结果是标量：

$$\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}.$$

手算时只做三件事：

1. 写出 $F_x, F_y, F_z$；
2. 第 $i$ 个分量只对第 $i$ 个坐标求导（$F_x$ 对 $x$，不要对 $y,z$ 求导后再加进去）；
3. 三项相加。没有减号，也不要和旋度的行列式混用。

二维情形丢掉 $z$ 分量即可：$\nabla \cdot \mathbf{F} = \partial F_x/\partial x + \partial F_y/\partial y$。

### 3.2 几何意义：单位体积的净流出

设 $V$ 是包含点 $\mathbf{x}_0$ 的一块小区域，$|V|$ 是它的体积。$\partial V$ 是 $V$ 的边界：这里的 $\partial$ 表示“取边界”，不是偏导数。则

$$(\nabla \cdot \mathbf{F})(\mathbf{x}_0) = \lim_{|V|\to 0} \frac{1}{|V|} \oiint_{\partial V} \mathbf{F}\cdot d\mathbf{S}.$$

$\oiint$ 是封闭曲面上的积分。$d\mathbf{S}=\mathbf{n}\,dS$ 是外法向面积元：$\mathbf{n}$ 是指向区域外部的单位法向量，$dS$ 是面积微元。右边是“穿出封闭曲面的通量，再除以体积”。所以散度是 **源汇强度的体密度**：

| 符号 | 通量 | 称呼 | 图像 |
|------|------|------|------|
| $\nabla \cdot \mathbf{F} > 0$ | 净流出 | 源 | 从该点向外冒 |
| $\nabla \cdot \mathbf{F} < 0$ | 净流入 | 汇 | 向该点汇集 |
| $\nabla \cdot \mathbf{F} = 0$ | 局部平衡 | 无源（管状场） | 流进多少就流出多少 |

$\nabla \cdot \mathbf{F} = 0$ 只说明无穷小体积上通量为零，不表示场本身为零。纯旋转场 $\mathbf{F} = (-y, x, 0)$ 的散度就是 $0$，但场并不静止。

### 3.3 散度定理

$V$ 是边界分片光滑的有界区域，$\mathbf{F}$ 在 $V$ 上连续可微，$dV$ 是体积元时，

$$\iiint_V (\nabla \cdot \mathbf{F})\,dV = \oiint_{\partial V} \mathbf{F}\cdot d\mathbf{S}.$$

$\iiint$ 是三重积分。左边把体内每一点的源加总，右边是边界上的总流出。第六节用单位立方体把两边分别算出，用来确认散度没有算错。

定理对含奇点的场不能直接套在包含奇点的区域上。$\mathbf{F} = \mathbf{r}/r^3$ 在原点之外散度为 $0$，但包围原点的任一闭曲面通量都是 $4\pi$。原点必须单独处理，见 6.4 节。

### 3.4 在方程里表示什么

| 方程 | 散度所在的位置 | 读法 |
|------|----------------|------|
| 连续性方程 $\partial_t \rho + \nabla\cdot(\rho\mathbf{v}) = 0$ | 质量通量的散度 | 密度增加，当且仅当有质量净流入 |
| 高斯定律 $\nabla\cdot\mathbf{E} = \rho_e/\varepsilon_0$ | 电场的散度 | 电荷就是电场的源 |
| 不可压缩 $\nabla\cdot\mathbf{v} = 0$ | 速度的散度 | 流体微团体积不变 |
| 磁高斯定律 $\nabla\cdot\mathbf{B} = 0$ | 磁场的散度 | 没有磁单极，磁力线闭合 |

$\partial_t$ 是 $\partial/\partial t$，$t$ 是时间。$\rho$ 是质量密度，$\mathbf{v}$ 是速度。$\rho_e$ 是电荷密度，$\varepsilon_0$ 是真空介电常数，$\mathbf{B}$ 是磁场。连续性方程是散度最值得记住的一次出现：$\nabla\cdot(\rho\mathbf{v})>0$ 表示该点在向周围输送质量，当地密度因此下降。

### 3.5 乘积法则与常用恒等式

以下 $f$ 是标量场，$\mathbf{A}$、$\mathbf{B}$ 是任意向量场。这里的 $\mathbf{B}$ 与上一小节的磁场不是同一个对象。

$$\nabla \cdot (f\mathbf{A}) = \nabla f \cdot \mathbf{A} + f\,(\nabla \cdot \mathbf{A}).$$

第一项是标量变化造成的通量，第二项是向量场本身的源再被 $f$ 放大。

两个向量场的叉乘：

$$\nabla \cdot (\mathbf{A}\times\mathbf{B}) = \mathbf{B}\cdot(\nabla\times\mathbf{A}) - \mathbf{A}\cdot(\nabla\times\mathbf{B}).$$

由此立刻得到一条后面要反复用的结论：任何旋度的散度都是零（令上式里的场来自同一个势，或直接对分量求偏导），

$$\nabla \cdot (\nabla \times \mathbf{A}) = 0.$$

只要区域合适，散度为零的场可以写成某个向量势的旋度。对上一小节的磁场，$\nabla\cdot\mathbf{B}=0$ 时存在磁矢势 $\mathbf{A}$ 使 $\mathbf{B}=\nabla\times\mathbf{A}$。

### 3.6 柱坐标与球坐标

直角坐标之外，散度 **不是** 把三个分量的偏导相加。尺度因子会多出一些零阶项。下面 $\rho$ 是柱径向坐标（不要和上一小节的质量密度混用），$\varphi$ 是绕 $z$ 轴的方位角；球坐标里 $r$ 是到原点的距离，$\theta$ 是从 $z$ 轴正方向量起的极角。分量下标表示沿该方向的投影，不是求导：例如 $A_\rho$ 是 $\mathbf{A}$ 的柱径向分量。单位向量统一写成 $\mathbf{e}$ 加上下标，$\mathbf{e}_r=\mathbf{r}/r$ 是指向远离原点的径向单位向量。

柱坐标 $(\rho,\varphi,z)$：

$$\nabla\cdot\mathbf{A} = \frac{1}{\rho}\frac{\partial}{\partial\rho}(\rho A_\rho) + \frac{1}{\rho}\frac{\partial A_\varphi}{\partial\varphi} + \frac{\partial A_z}{\partial z}.$$

球坐标 $(r,\theta,\varphi)$：

$$\nabla\cdot\mathbf{A} = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 A_r) + \frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta\, A_\theta) + \frac{1}{r\sin\theta}\frac{\partial A_\varphi}{\partial\varphi}.$$

只依赖 $r$ 的径向场 $\mathbf{A} = A_r(r)\,\mathbf{e}_r$ 因此简化为

$$\nabla\cdot\mathbf{A} = \frac{1}{r^2}\frac{d}{dr}(r^2 A_r).$$

$\mathbf{A} = \mathbf{r}/r^3 = r^{-2}\mathbf{e}_r$ 时，$r^{-2}$ 表示 $1/r^2$，于是 $r^2 A_r = 1$，导数为零，在 $r\neq 0$ 处散度为 $0$。这和 3.3 节的通量并不矛盾：原点不在这个表达式的定义域里。

---

## 四、旋度 $\nabla \times \mathbf{F}$

### 4.1 定义与计算

旋度是 ∇ 与向量场的叉乘，结果仍是向量。直角坐标里写成行列式

$$\nabla\times\mathbf{F} = \begin{vmatrix} \mathbf{e}_x & \mathbf{e}_y & \mathbf{e}_z \\ \partial_x & \partial_y & \partial_z \\ F_x & F_y & F_z \end{vmatrix},$$

其中 $\mathbf{e}_x$、$\mathbf{e}_y$、$\mathbf{e}_z$ 是直角坐标的单位向量，分别指向 $x$、$y$、$z$ 轴的正方向，长度都是 $1$，并且两两垂直。$\partial_x$ 是前文的 $\partial/\partial x$，$\partial_y$、$\partial_z$ 同理；这一行是算符，只作用在第三行的分量上，不能与 $F_x,F_y,F_z$ 对调。

展开后的分量是

$$
\begin{aligned}
(\nabla\times\mathbf{F})_x &= \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z},\\[4pt]
(\nabla\times\mathbf{F})_y &= \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x},\\[4pt]
(\nabla\times\mathbf{F})_z &= \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}.
\end{aligned}
$$

$y$ 分量最容易写反。行列式按第二列展开时自带一个负号，所以是 $\partial F_x/\partial z - \partial F_z/\partial x$，而不是相反。三个分量按 $x\to y\to z\to x$ 轮换：每一项都是“对下一个坐标求偏导、作用在再下一个分量上”，再减去坐标对调后的那一项。

二维向量场 $(F_x, F_y)$ 的旋度只剩一个标量，就是三维公式里的 $z$ 分量：

$$(\nabla\times\mathbf{F})_z = \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}.$$

它也是格林公式里曲线积分与二重积分之间的那个被积函数。

### 4.2 几何意义：单位面积的环量

取一块过点 $\mathbf{x}_0$、单位法向量为 $\mathbf{n}$ 的小曲面 $S$，$|S|$ 是它的面积，边界 $\partial S$ 的走向与 $\mathbf{n}$ 成右手系。则

$$\mathbf{n}\cdot(\nabla\times\mathbf{F})(\mathbf{x}_0) = \lim_{|S|\to 0}\frac{1}{|S|}\oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}.$$

$\oint$ 是沿封闭曲线的积分。左边是旋度在法向上的投影，右边是 **单位面积上的环量**。因此：

- 旋度的方向是局部旋转轴，指向由右手定则决定；
- 旋度的大小表示绕该轴旋转的强弱；
- $\nabla\times\mathbf{F}=\mathbf{0}$ 表示每一个无穷小回路的环量都是 $0$，称场 **无旋**。

刚体旋转给出一个干净的比例。角速度为常向量 $\boldsymbol{\omega}$ 时，线速度 $\mathbf{v}=\boldsymbol{\omega}\times\mathbf{r}$，可以验证

$$\nabla\times\mathbf{v} = 2\boldsymbol{\omega}.$$

流体力学把涡量定义成 $\boldsymbol{\omega}_{\text{涡}}=\nabla\times\mathbf{v}$，所以刚体转动的涡量是角速度的两倍。第六节用 $\boldsymbol{\omega}=\mathbf{e}_z$ 把这个因子 $2$ 算出来。

### 4.3 斯托克斯定理

$S$ 是分片光滑的定向曲面，边界 $\partial S$ 与法向成右手系，$\mathbf{F}$ 在包含 $S$ 的开集上连续可微时，

$$\iint_S (\nabla\times\mathbf{F})\cdot d\mathbf{S} = \oint_{\partial S} \mathbf{F}\cdot d\mathbf{r}.$$

$\iint$ 是曲面积分。左边是旋度穿过曲面的通量，右边是边界环量。散度定理把体积分换成封闭曲面通量，斯托克斯定理把面积分换成边界线积分，两者地位相当。

### 4.4 无旋、保守场与单连通

下面四句话在 **单连通** 区域（例如整个 $\mathbb{R}^3$，即三维实空间，或球、立方体内部）上等价：

1. $\nabla\times\mathbf{F}=\mathbf{0}$；
2. 任意闭曲线上 $\oint\mathbf{F}\cdot d\mathbf{r}=0$；
3. 线积分与路径无关；
4. 存在标量势 $\varphi$，使 $\mathbf{F}=\nabla\varphi$（或按力场习惯写成 $\mathbf{F}=-\nabla\varphi$）。

第 4 条的势可以沿路径积分求出来。固定基点 $\mathbf{x}_0$，令

$$\varphi(\mathbf{x}) = \varphi(\mathbf{x}_0) + \int_{\mathbf{x}_0}^{\mathbf{x}} \mathbf{F}\cdot d\mathbf{r}.$$

无旋保证这个积分不依赖路径。

区域若不单连通，无旋 **不能** 推出全局势函数。经典反例是去掉 $z$ 轴之后的场

$$\mathbf{F} = \left(-\frac{y}{x^2+y^2},\ \frac{x}{x^2+y^2},\ 0\right).$$

在 $x^2+y^2\neq 0$ 处 $\nabla\times\mathbf{F}=\mathbf{0}$，但绕 $z$ 轴一周的环量是 $2\pi$。局部仍然有势 $\varphi=\arg(x+iy)$。$\arg$ 是复数 $x+iy$ 的辐角，也就是从正 $x$ 轴转到 $(x,y)$ 的有向角；它绕轴一圈会增加 $2\pi$，不能在整条轴被挖掉的空间里做成单值函数。计算过程见 6.6 节。

梯度的旋度恒为零，和上面的判定一致：

$$\nabla\times(\nabla f) = \mathbf{0}.$$

只要 $f$ 的二阶混合偏导连续，施瓦茨定理让每一对交叉导数相减得零。因此“先做梯度、再做旋度”一定得到零场；“先做旋度、再做散度”也一定得到零标量。

### 4.5 在方程里表示什么

| 方程 | 读法 |
|------|------|
| 法拉第定律 $\nabla\times\mathbf{E} = -\partial_t\mathbf{B}$ | 变化的磁场产生环状电场 |
| 安培–麦克斯韦定律 $\nabla\times\mathbf{B}=\mu_0\mathbf{J}+\mu_0\varepsilon_0\partial_t\mathbf{E}$ | 电流与变化的电场产生环状磁场 |
| 无旋条件 $\nabla\times\mathbf{E}=\mathbf{0}$（静电场） | 静电场可以写成 $-\nabla\varphi$ |
| 涡量输运中的 $\boldsymbol{\omega}=\nabla\times\mathbf{v}$ | 速度场的局部旋转 |

$\mu_0$ 是真空磁导率，$\mathbf{J}$ 是电流密度。$\boldsymbol{\omega}$ 在这一行是涡量；上文刚体转动里的 $\boldsymbol{\omega}$ 是角速度，二者差一个因子 $2$。静电场无旋，所以用一个标量电位就够了。稳恒磁场无散，所以用一个向量势来写。两个势的存在性，分别就是本节和上一节的两条恒等式。

### 4.6 乘积法则

$$\nabla\times(f\mathbf{A}) = \nabla f\times\mathbf{A} + f\,(\nabla\times\mathbf{A}).$$

$\nabla f\times\mathbf{A}$ 是新项：标量的梯度若与 $\mathbf{A}$ 不平行，会额外贡献旋度。即便 $\mathbf{A}$ 本身无旋，$f\mathbf{A}$ 仍可能有旋。

二重旋度把旋度和拉普拉斯连起来。在直角坐标中

$$\nabla\times(\nabla\times\mathbf{A}) = \nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A},$$

其中 $\nabla^2\mathbf{A}$ 表示对每个直角分量分别做标量拉普拉斯。移项就得到向量泊松方程、粘性项和电磁波动方程里常见的分解：无散部分由旋度控制，无旋部分由散度控制。

### 4.7 柱坐标与球坐标

柱坐标 $(\rho,\varphi,z)$：

$$
\begin{aligned}
(\nabla\times\mathbf{A})_\rho &= \frac{1}{\rho}\frac{\partial A_z}{\partial\varphi} - \frac{\partial A_\varphi}{\partial z},\\[4pt]
(\nabla\times\mathbf{A})_\varphi &= \frac{\partial A_\rho}{\partial z} - \frac{\partial A_z}{\partial\rho},\\[4pt]
(\nabla\times\mathbf{A})_z &= \frac{1}{\rho}\frac{\partial}{\partial\rho}(\rho A_\varphi) - \frac{1}{\rho}\frac{\partial A_\rho}{\partial\varphi}.
\end{aligned}
$$

球坐标 $(r,\theta,\varphi)$：

$$
\begin{aligned}
(\nabla\times\mathbf{A})_r &= \frac{1}{r\sin\theta}\left(\frac{\partial}{\partial\theta}(\sin\theta\, A_\varphi) - \frac{\partial A_\theta}{\partial\varphi}\right),\\[4pt]
(\nabla\times\mathbf{A})_\theta &= \frac{1}{r\sin\theta}\frac{\partial A_r}{\partial\varphi} - \frac{1}{r}\frac{\partial}{\partial r}(r A_\varphi),\\[4pt]
(\nabla\times\mathbf{A})_\varphi &= \frac{1}{r}\frac{\partial}{\partial r}(r A_\theta) - \frac{1}{r}\frac{\partial A_r}{\partial\theta}.
\end{aligned}
$$

曲线坐标里不要把标量拉普拉斯直接套到各个分量上再组装成向量拉普拉斯。直角坐标可以这样做，柱坐标和球坐标不行。曲线坐标下的向量拉普拉斯应通过 $\nabla(\nabla\cdot\mathbf{A})-\nabla\times(\nabla\times\mathbf{A})$ 来算。

---

## 五、拉普拉斯算子 $\nabla^2$

### 5.1 定义

$\nabla^2$ 也写作 $\Delta$，是 ∇ 与自身的点乘，也就是“先取梯度，再取散度”：

$$\nabla^2 f = \nabla\cdot(\nabla f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}.$$

它把标量场送回标量场。作用在直角坐标的向量场上时，对每个分量各做一次：

$$\nabla^2\mathbf{F} = (\nabla^2 F_x,\ \nabla^2 F_y,\ \nabla^2 F_z).$$

纳维–斯托克斯方程的粘性项 $\nu\nabla^2\mathbf{u}$ 就是这种分量形式（在直角坐标中）。这里 $\mathbf{u}$ 是流体速度，$\nu$ 是运动粘度。后面方程表里的标量 $u$ 是另一个未知函数，不是这个速度。

### 5.2 和周围平均值的关系

设 $n$ 是空间维数。$B_r$ 是半径为 $r$ 的球，$\partial B_r$ 是这个球的球面，$|\partial B_r|$ 是球面面积。$f$ 在球面上的平均值与中心值之差满足

$$\frac{1}{|\partial B_r|}\int_{\partial B_r} f\,dS - f(\mathbf{x}) = \frac{r^2}{2n}\,\nabla^2 f(\mathbf{x}) + O(r^4).$$

$O(r^4)$ 表示余项的大小不超过某个常数乘 $r^4$。$r$ 趋于 $0$ 时，它比右边的 $r^2$ 项更快地消失。

系数 $r^2/(2n)$ 为正，因此：

- $\nabla^2 f>0$：周围平均高于该点，该点是局部的凹谷；
- $\nabla^2 f<0$：周围平均低于该点，该点是局部的凸峰；
- $\nabla^2 f=0$：该点等于周围平均值（精确到四阶小量；调和函数则精确相等）。

热传导里这个符号直接决定温度升降。$\partial_t u = \alpha\nabla^2 u$ 且 $\alpha>0$ 时，比周围冷的点（$\nabla^2 u>0$）会被加热，比周围热的点会变冷。

$\nabla^2 f=0$ 的函数称为调和函数。调和函数满足平均值性质：球心的值等于球面平均值，内部的最大值和最小值都在边界上取到。

### 5.3 方程一览

| 方程 | 形式 | 含义 |
|------|------|------|
| 拉普拉斯方程 | $\nabla^2 u = 0$ | 稳态、该区域内无源 |
| 泊松方程 | $\nabla^2 u = \rho$ | 稳态、有源。右端 $\rho$ 表示源，不是第三节的质量密度；源的符号吸收在它的定义里 |
| 热传导方程 | $\partial_t u = \alpha\nabla^2 u$ | $\alpha$ 是热扩散系数；$\alpha>0$ 时温度向周围平均靠拢 |
| 波动方程 | $\partial_{tt} u = c^2\nabla^2 u$ | $c$ 是波速 |
| 薛定谔方程 | $\displaystyle i\hbar\partial_t\psi = -\frac{\hbar^2}{2m}\nabla^2\psi + V\psi$ | 动能项来自拉普拉斯算子 |

表中的 $u$ 是未知标量函数。$\partial_{tt}$ 是 $\partial^2/\partial t^2$。薛定谔方程里，$i$ 是虚数单位，满足 $i^2=-1$；$\hbar=h/(2\pi)$ 是约化普朗克常数，$h$ 是普朗克常数；$\psi$ 是波函数，$m$ 是粒子质量，$V$ 是势能。这个 $V$ 与第三节里表示区域的 $V$ 不是同一个符号。

静电场若写成 $\mathbf{E}=-\nabla\varphi$，高斯定律 $\nabla\cdot\mathbf{E}=\rho_e/\varepsilon_0$ 会变成

$$\nabla^2\varphi = -\frac{\rho_e}{\varepsilon_0}.$$

这里的负号来自 $\mathbf{E}=-\nabla\varphi$，不是拉普拉斯算子本身又变了号。表中的泊松方程把这个符号放进了右端项 $\rho$。

### 5.4 径向公式与基本解

球坐标中只依赖 $r$ 的函数：

$$\nabla^2 u(r) = \frac{1}{r^2}\frac{d}{dr}\!\left(r^2\frac{du}{dr}\right) = u'' + \frac{2}{r}u'.$$

一撇 $u'$ 表示 $du/dr$，两撇 $u''$ 表示 $d^2u/dr^2$。

$u=1/r$ 时 $r^2 u' = -1$，再求导为零，所以

$$\nabla^2\!\left(\frac{1}{r}\right) = 0 \qquad (r\neq 0).$$

把原点补回去之后，在分布意义下

$$\nabla^2\!\left(\frac{1}{r}\right) = -4\pi\delta(\mathbf{r}).$$

$\delta(\mathbf{r})$ 是三维狄拉克 $\delta$：原点以外等于 $0$，对全空间的积分等于 $1$。它不是普通函数，只能放在积分里理解；前面说的“分布意义”就是这个意思。二维对应物是 $u=\ln\rho$，$\rho=\sqrt{x^2+y^2}$：原点之外调和，而 $\nabla^2\ln\rho = 2\pi\delta$。这里的 $\delta$ 是二维狄拉克 $\delta$，对整个平面的积分等于 $1$。三维通量的算法见 6.4 节。

三维柱坐标与球坐标的一般表达式：

$$
\begin{aligned}
\nabla^2 f &= \frac{1}{\rho}\frac{\partial}{\partial\rho}\!\left(\rho\frac{\partial f}{\partial\rho}\right) + \frac{1}{\rho^2}\frac{\partial^2 f}{\partial\varphi^2} + \frac{\partial^2 f}{\partial z^2},\\[6pt]
\nabla^2 f &= \frac{1}{r^2}\frac{\partial}{\partial r}\!\left(r^2\frac{\partial f}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\!\left(\sin\theta\frac{\partial f}{\partial\theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 f}{\partial\varphi^2}.
\end{aligned}
$$

### 5.5 三条把四个算子串起来的恒等式

$$
\begin{aligned}
\nabla\times(\nabla f) &= \mathbf{0},\\[4pt]
\nabla\cdot(\nabla\times\mathbf{A}) &= 0,\\[4pt]
\nabla\times(\nabla\times\mathbf{A}) &= \nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A}.
\end{aligned}
$$

前两条说明：梯度场无旋，旋度场无散。第三条在已知散度和旋度时，把向量场的拉普拉斯算子解出来。赫姆霍兹分解把这两条用到底：适当衰减的光滑向量场可以唯一地拆成

$$\mathbf{F} = -\nabla\varphi + \nabla\times\mathbf{A},$$

第一项无旋，第二项无散。很多线性偏微分方程先分别对 $\varphi$ 和 $\mathbf{A}$ 求解，再合成 $\mathbf{F}$。

---

## 六、实际算例

这一节的导数都按定义展开。算完后优先用恒等式做检查：梯度的旋度应为 $\mathbf{0}$，旋度的散度应为 $0$，散度定理两边应给出同一个数。

### 6.1 梯度：方向、等值面、方向导数

取

$$f(x,y,z) = xy^2 + z^3.$$

一阶偏导

$$\frac{\partial f}{\partial x} = y^2,\qquad \frac{\partial f}{\partial y} = 2xy,\qquad \frac{\partial f}{\partial z} = 3z^2,$$

所以

$$\nabla f = (y^2,\ 2xy,\ 3z^2).$$

在 $P=(1,2,-1)$ 处

$$\nabla f(P) = (4,\ 4,\ 3).$$

$|\nabla f(P)| = \sqrt{16+16+9}=\sqrt{41}$，这就是 $f$ 在 $P$ 处的最大上升率，方向为 $(4,4,3)$。

再取单位向量 $\mathbf{u}=(1,1,0)/\sqrt{2}$。方向导数

$$D_{\mathbf{u}}f(P) = (4,4,3)\cdot\frac{(1,1,0)}{\sqrt{2}} = \frac{8}{\sqrt{2}} = 4\sqrt{2}.$$

它小于 $\sqrt{41}\approx 6.40$，符合“方向导数不超过梯度模长”。

等值面 $f=c$ 在 $P$ 处的法向就是 $(4,4,3)$。切平面上任一方向都与 $(4,4,3)$ 垂直。

### 6.2 梯度：距离函数

设 $r=\sqrt{x^2+y^2+z^2}$。由链式法则

$$\frac{\partial r}{\partial x} = \frac{x}{r}\quad(r\neq 0),$$

另外两个分量类似，因此 $\nabla r = \mathbf{r}/r$，即单位径向向量。再对 $1/r$：

$$\frac{\partial}{\partial x}r^{-1} = -r^{-2}\frac{\partial r}{\partial x} = -\frac{x}{r^3},$$

故

$$\nabla\!\left(\frac{1}{r}\right) = -\frac{\mathbf{r}}{r^3}\qquad(r\neq 0).$$

静电场点电荷势常取成与 $1/r$ 成正比，记作 $\varphi \propto 1/r$；符号 $\propto$ 读作“正比于”。电场 $\mathbf{E}=-\nabla\varphi$ 就沿着 $\mathbf{r}/r^3$ 的方向。

### 6.3 散度：源的强弱，并用散度定理核对

取 $\mathbf{F}=(2x,\ 3y,\ -z)$。

$$\nabla\cdot\mathbf{F} = 2 + 3 + (-1) = 4.$$

各点源强相同，都是 $4$。在单位立方体 $V=[0,1]^3$ 上

$$\iiint_V 4\,dV = 4.$$

再用定义计算外法向通量。只有外法向指向坐标增加的那三个面有贡献：

| 面 | 外法向 | $\mathbf{F}\cdot\mathbf{n}$ | 通量 |
|----|--------|------------------------------|------|
| $x=1$ | $+\mathbf{e}_x$ | $2$ | $2$ |
| $x=0$ | $-\mathbf{e}_x$ | $0$ | $0$ |
| $y=1$ | $+\mathbf{e}_y$ | $3$ | $3$ |
| $y=0$ | $-\mathbf{e}_y$ | $0$ | $0$ |
| $z=1$ | $+\mathbf{e}_z$ | $-1$ | $-1$ |
| $z=0$ | $-\mathbf{e}_z$ | $0$ | $0$ |

六个面相加仍是 $4$，与体积分一致。$z=1$ 这一面通量为负，是因为 $F_z=-z$ 在顶面指向体内，这一面是汇而不是源；三个方向合起来仍然是净源。

作为对照，纯旋转场 $\mathbf{v}=(-y,\ x,\ 0)$ 的散度

$$\nabla\cdot\mathbf{v} = \frac{\partial}{\partial x}(-y) + \frac{\partial}{\partial y}(x) + \frac{\partial}{\partial z}(0) = 0.$$

场在转，但没有源。它的旋度留到 6.5 节。

### 6.4 散度：原点上的点源

设 $\mathbf{F}=\mathbf{r}/r^3$，即 $F_x = x\,r^{-3}$。负指数 $r^{-3}$ 就是 $1/r^3$，其余分量类似。$r\neq 0$ 时

$$\frac{\partial F_x}{\partial x} = r^{-3} - 3x^2 r^{-5},$$

$y,z$ 分量同样计算后相加：

$$\nabla\cdot\mathbf{F} = 3r^{-3} - 3(x^2+y^2+z^2)r^{-5} = 3r^{-3} - 3r^{-3} = 0.$$

原点之外没有源。再看以原点为心、半径 $a$ 的球面：$\mathbf{F}\cdot\mathbf{e}_r = 1/a^2$。面积元写成 $a^2\,d\Omega$，其中 $d\Omega$ 是立体角微元；整个球面的立体角是 $4\pi$，所以积分下标记成 $4\pi$：

$$\oiint \mathbf{F}\cdot d\mathbf{S} = \int_{4\pi} 1\,d\Omega = 4\pi.$$

通量与半径无关。散度定理不能直接用在包含原点的球上，因为 $\mathbf{F}$ 在原点不可微。分布意义下把这 $4\pi$ 的通量记成原点处的源：

$$\nabla\cdot\frac{\mathbf{r}}{r^3} = 4\pi\delta(\mathbf{r}).$$

结合 6.2 节的梯度，$\nabla(1/r)=-\mathbf{r}/r^3$，两边再取散度：

$$\nabla^2\!\left(\frac{1}{r}\right) = -4\pi\delta(\mathbf{r}).$$

这就是三维泊松方程 $-\nabla^2 \Phi = 4\pi\delta$ 的一个基本解 $\Phi=1/r$。这里 $\Phi$ 是势函数，$\delta$ 是 5.4 节的三维狄拉克 $\delta$。计算普通导数时先避开原点，再用封闭曲面上的通量把原点的贡献补上。

### 6.5 旋度：刚体旋转

角速度 $\boldsymbol{\omega}=(0,0,1)$ 对应的线速度

$$\mathbf{v}=\boldsymbol{\omega}\times\mathbf{r}=(-y,\ x,\ 0).$$

代入旋度公式：

$$
\begin{aligned}
(\nabla\times\mathbf{v})_x &= \partial_y(0) - \partial_z(x) = 0,\\
(\nabla\times\mathbf{v})_y &= \partial_z(-y) - \partial_x(0) = 0,\\
(\nabla\times\mathbf{v})_z &= \partial_x(x) - \partial_y(-y) = 1-(-1) = 2.
\end{aligned}
$$

所以 $\nabla\times\mathbf{v}=(0,0,2)=2\boldsymbol{\omega}$。散度在 6.3 节已经是 $0$。这是一个无源、但处处以同一角速度旋转的场。

用斯托克斯定理核对。取 $xy$ 平面上的单位圆盘 $S$，法向 $\mathbf{e}_z$，边界为单位圆 $\mathbf{r}=(\cos t,\sin t,0)$。$t$ 是绕行参数，$t\in[0,2\pi]$ 表示逆时针走一整圈。

面积分：

$$\iint_S (\nabla\times\mathbf{v})\cdot d\mathbf{S} = \iint_S 2\,dA = 2\pi.$$

线积分：$d\mathbf{r}=(-\sin t,\ \cos t,\ 0)\,dt$，而在单位圆上 $\mathbf{v}=(-\sin t,\ \cos t,\ 0)$，

$$\mathbf{v}\cdot d\mathbf{r} = (\sin^2 t+\cos^2 t)\,dt = dt, \qquad \oint\mathbf{v}\cdot d\mathbf{r} = \int_0^{2\pi} dt = 2\pi.$$

两边一致。

### 6.6 旋度：无旋却没有全局势

考虑挖去 $z$ 轴后的场

$$\mathbf{F}=\left(-\frac{y}{x^2+y^2},\ \frac{x}{x^2+y^2},\ 0\right).$$

记 $\rho^2=x^2+y^2$。直接求 $z$ 分量：

$$
\begin{aligned}
\frac{\partial F_y}{\partial x} &= \frac{\rho^2-2x^2}{\rho^4} = \frac{y^2-x^2}{\rho^4},\\[4pt]
\frac{\partial F_x}{\partial y} &= \frac{-\rho^2+2y^2}{\rho^4} = \frac{y^2-x^2}{\rho^4}.
\end{aligned}
$$

二者相减为 $0$。$x,y$ 分量里对 $z$ 的导数都是 $0$，所以在 $\rho\neq 0$ 处 $\nabla\times\mathbf{F}=\mathbf{0}$。散度同样为 $0$。

再沿单位圆 $\mathbf{r}=(\cos t,\sin t,0)$ 积分。圆上 $\mathbf{F}=(-\sin t,\ \cos t,\ 0)$，$d\mathbf{r}=(-\sin t,\ \cos t,\ 0)\,dt$，

$$\oint\mathbf{F}\cdot d\mathbf{r} = \int_0^{2\pi} 1\,dt = 2\pi \neq 0.$$

闭路积分不为零，所以在“空间去掉整条 $z$ 轴”这个区域上不存在单值势。若把平面沿正 $x$ 轴南侧剪开，使区域变成单连通，则 $\varphi=\operatorname{atan2}(y,x)$ 满足 $\mathbf{F}=\nabla\varphi$。$\operatorname{atan2}(y,x)$ 是带象限的辐角，由 $x,y$ 的符号决定落在哪一支，取值通常在 $(-\pi,\pi]$；它跨过剪开处有 $2\pi$ 的跳跃。

这个例子用来卡住一个常见判断：只算出 $\nabla\times\mathbf{F}=\mathbf{0}$ 还不够，还要看区域是不是单连通。

### 6.7 同时计算散度与旋度

取 $\mathbf{F}=(x^2 y,\ yz,\ xz)$。

散度只保留“分量对自身坐标”的导数：

$$\nabla\cdot\mathbf{F} = \frac{\partial}{\partial x}(x^2 y) + \frac{\partial}{\partial y}(yz) + \frac{\partial}{\partial z}(xz) = 2xy + z + x.$$

旋度：

$$
\begin{aligned}
(\nabla\times\mathbf{F})_x &= \partial_y(xz)-\partial_z(yz) = 0-y = -y,\\
(\nabla\times\mathbf{F})_y &= \partial_z(x^2 y)-\partial_x(xz) = 0-z = -z,\\
(\nabla\times\mathbf{F})_z &= \partial_x(yz)-\partial_y(x^2 y) = 0-x^2 = -x^2.
\end{aligned}
$$

即 $\nabla\times\mathbf{F}=(-y,\ -z,\ -x^2)$。再对旋度取散度做检查：

$$\nabla\cdot(-y,\ -z,\ -x^2) = 0+0+0=0.$$

符合 $\nabla\cdot(\nabla\times\mathbf{F})=0$。由于旋度不是零场，$\mathbf{F}$ 不是任何标量的梯度。

### 6.8 拉普拉斯，以及带进方程里验算

**调和函数。** $u=x^2-y^2$ 时

$$\nabla^2 u = 2 + (-2) = 0.$$

$u$ 是马鞍面，中心既不是峰也不是谷，拉普拉斯算子给出 $0$，与 5.2 节的平均值图像一致。

二维对数：$u=\ln(x^2+y^2)=2\ln\rho$（$\rho\neq 0$）。

$$\frac{\partial u}{\partial x}=\frac{2x}{\rho^2},\qquad \frac{\partial^2 u}{\partial x^2}=\frac{2(\rho^2-2x^2)}{\rho^4}=\frac{2(y^2-x^2)}{\rho^4},$$

$\partial^2 u/\partial y^2$ 把 $x,y$ 对调，两者相加为 $0$。因此去掉原点之后 $u$ 调和。因为 $\nabla^2\ln\rho=2\pi\delta$，这里 $u=2\ln\rho$，原点处的分布是 $4\pi\delta$。

**热方程。** 取 $u(x,y,t)=e^{-2t}\sin x\sin y$。

$$
\begin{aligned}
\partial_t u &= -2e^{-2t}\sin x\sin y,\\
\partial_{xx}u &= -e^{-2t}\sin x\sin y,\\
\partial_{yy}u &= -e^{-2t}\sin x\sin y.
\end{aligned}
$$

于是 $\nabla^2 u = -2e^{-2t}\sin x\sin y = \partial_t u$。$u$ 满足 $\partial_t u = \nabla^2 u$，即热传导方程中 $\alpha=1$ 的一个解。指数上的 $2$ 来自两个空间方向各贡献一个 $-1$。

**波动方程。** 取 $u(x,t)=\cos x\cos(ct)$。则 $\partial_{tt}u=-c^2\cos x\cos(ct)$，$\partial_{xx}u=-\cos x\cos(ct)$，所以

$$\partial_{tt}u = c^2\partial_{xx}u = c^2\nabla^2 u.$$

这是一维波动方程的驻波解。换成 $\cos(\mathbf{k}\cdot\mathbf{r}-\omega t)$ 时，同样的算法要求 $\omega^2=c^2|\mathbf{k}|^2$。这里 $\mathbf{k}$ 是波矢，$|\mathbf{k}|$ 是它的长度，$\omega$ 是角频率；这个 $\omega$ 与第四节里的角速度不是同一个量。

### 6.9 恒等式：梯度的旋度为零

取 $f=x^2 yz+\sin(y+z)$。梯度

$$\nabla f = \big(2xyz,\ x^2 z+\cos(y+z),\ x^2 y+\cos(y+z)\big).$$

旋度的 $x$ 分量

$$\partial_y(x^2 y+\cos(y+z)) - \partial_z(x^2 z+\cos(y+z)) = \big(x^2-\sin(y+z)\big)-\big(x^2-\sin(y+z)\big)=0.$$

$y$ 分量：$\partial_z(2xyz)-\partial_x(x^2 y+\cos(y+z))=2xy-2xy=0$。

$z$ 分量：$\partial_x(x^2 z+\cos(y+z))-\partial_y(2xyz)=2xz-2xz=0$。

因此 $\nabla\times(\nabla f)=\mathbf{0}$。用这个办法检查势函数是否求对：求出候选的 $\varphi$ 之后再算一次梯度，应回到原场。

## 附录：记号、关系图与速记

### A. 带箭头的 $\vec{\nabla}$

$\vec{\nabla}$ 和 $\nabla$ 是同一个算子。箭头只是记号上的提醒，不改变任何计算结果。

- 带箭头时，作者希望读者把它当作向量：可以点乘（得到散度）、叉乘（得到旋度），也可以放进向量恒等式。
- 少数教材用没有箭头的 $\nabla$ 泛指“微分算子”这个符号，用 $\vec{\nabla}$ 强调它正以向量形式参与运算。
- 阅读和手算时，两种写法都按本文的梯度、散度、旋度处理。

### B. 整体关系

```text
∇   Nabla / Del
├── ∇f            梯度      标量 → 向量     最陡上升方向与速率
├── ∇ · F         散度      向量 → 标量     源 / 汇的体密度
├── ∇ × F         旋度      向量 → 向量     局部旋转轴与强弱
└── ∇² = ∇ · ∇    拉普拉斯  标量 → 标量     偏离周围平均的程度
      ├── ∇²u = 0           拉普拉斯方程
      ├── ∇²u = ρ           泊松方程
      ├── u_t = α ∇²u       热传导方程
      ├── u_tt = c² ∇²u     波动方程
      └── 薛定谔方程中的动能项

恒等式：  ∇×(∇f) = 0 ，  ∇·(∇×F) = 0
记号：    箭头 ∇ 与无箭头 ∇ 相同，只强调按向量做点乘和叉乘
```

### C. 一句话速记

- **∇**：带着偏导的形式向量。贴在标量前面是梯度，点乘向量是散度，叉乘向量是旋度。
- **散度**：看一小块体积的净流出。正是源，负是汇，零是局部进出平衡。
- **旋度**：看一小块曲面的环量。方向沿转轴，刚体转动时它等于角速度的两倍；它为零只在单连通区域上保证存在势。
- **$\vec{\nabla}$**：多一个箭头，表示“按向量来运算”，数值上没有新内容。
- **∇²**：拿该点和周围平均值比。大于零，该点偏低；小于零，该点偏高；等于零，已经调和。

### D. 计算时最容易错的三处

1. 旋度的 $y$ 分量是 $\partial_z F_x-\partial_x F_z$，减号在第二项。展开行列式时不要把三行都写成“前减后”的同一种顺序。
2. 柱坐标、球坐标的散度要先乘尺度因子再求导，不能把 $\partial_\rho A_\rho+\partial_\varphi A_\varphi+\partial_z A_z$ 当成答案。
3. $\nabla\times\mathbf{F}=\mathbf{0}$ 要加上区域单连通，才能写出全局势。挖去一条轴、一个圈，都可能让闭路积分不为零。
