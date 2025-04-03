# 各种导数

## softmax交叉熵

### forward 

```math
crossentropy=-\log\frac{e^{Z_{target}-max({{Z}_i})}}{\sum_{i=1}^n e^{Z_i-max({{Z}_i})}}
```

### backward

这里我们关注Zi变化��Loss的影响，可以看出，当i不等于target时，Zi只作用于分母
反之则同时作用于分子分母，导数为作用于分子和分母的导数之和


当 $i \neq target$

令 $L=g_1(x_1)=-log(x_1)$

令 $x_1=g_2(x_2)=\frac{c1}{x_2}$

$c1=e^{Z_{target}-max({{Z}_i})}$

令 $x_2=g_3(x_3)=x_3+c2$

c2为常量

令 $x_3=g_4(x_4)=e^{x_4}$

令 $x_4=g_5(Z_i)=Z_i-max({{Z}_i})$

令 $sum=\sum_{i=1}^n e^{Z_i-max({{Z}_i})}$

故 $\frac{\partial L}{\partial {Z}_i}=\frac{\partial g_1(x_1)}{\partial x_1}\frac{\partial g_2(x_2)}{\partial x_2}\frac{\partial g_3(x_3)}{\partial x_3}\frac{\partial g_4(x_4)}{\partial x_4}\frac{\partial g_5(Z_i)}{\partial Z_i}$

$\frac{\partial g_1(x_1)}{\partial x_1}=-\frac{1}{x1}$

$\frac{\partial g_2(x_2)}{\partial x_2}=-\frac{c1}{x_2^2}$

$\frac{\partial g_3(x_3)}{\partial x_3}=1$

$\frac{\partial g_4(x_4)}{\partial x_4}=e^{x_4}$

$\frac{\partial g_5(Z_i)}{\partial Z_i}=1$

$x1=\frac{c1}{sum}$

$x2=sum$

$x_4=Z_i-max({{Z}_i})$

故

$\frac{\partial L}{\partial {Z}_i}=\frac{c1e^{x_4}}{x_1x_2^2}=\frac{e^{Z_i-max({{Z}_i})}}{sum}$


当 $i = target$

分母部分的导数同上

$\frac{e^{Z_{target}-max({{Z}_i})}}{sum}$

下面计算分子部分p

令 $g_1(x_1) = -log(x_1)$

令 $x_1 = g_2(x_2) = \frac{x_2}{sum}$

令 $x_2 = g_3(x_3) = e^{x_3}$

令 $x_3 = g_4(Z_{target})=Z_{target}-max({{Z}_i})$

$p=\frac{\partial g_1(x_1)}{\partial x_1}\frac{\partial g_2(x_2)}{\partial x_2}\frac{\partial g_3(x_3)}{\partial x_3}\frac{\partial g_4(Z_{target})}{\partial Z_{target}}$

$\frac{\partial g_1(x_1)}{\partial x_1}=-\frac{1}{x_1}$

$\frac{\partial g_2(x_2)}{\partial x_2}=\frac{1}{sum}$

$\frac{\partial g_3(x_3)}{\partial x_3}=e^{x_3}$

$\frac{\partial g_4(Z_{target})}{\partial Z_{target}}=1$

故 $p=-\frac{e^{x_3}}{x_1sum}$

$x_3=Z_{target}-max({{Z}_i})$

$x_1=\frac{e^{Z_{target}-max({{Z}_i})}}{sum}$

故 $p=-1$

故整体的导数为 $\frac{e^{Z_{target}-max({{Z}_i})}}{sum}-1$

```math
\frac{\partial ce}{\partial Z_i}=\begin{cases}\frac{e^{Z_i-max({{Z}_i})}}{sum}, & \text{if } i \neq target \\
\frac{e^{Z_{target}-max({{Z}_i})}}{sum}-1, & \text{if } i = target
\end{cases}
```

## softmax

### forward

$softmax(Z_i)=\frac{e^{Z_i}}{\sum_{j=1}^ne^{Z_j}}$

### backward

令 $softmax(Z_i) = g_1(x_1, x_2) = \frac{x_1}{x_2}$

令 $sum=\sum_{j=1}^ne^{Z_j}$

$\frac{\partial softmax(Z_i)}{\partial Z_i}=\frac{\partial g_1(x_1, x_2)}{\partial x_1}\frac{\partial x_1}{\partial Z_i}+\frac{\partial g_1(x_1, x_2)}{\partial x_2}\frac{\partial x_2}{\partial Z_i}$

同样考虑 i 是否等于 target的两种情况

当 $i=target$

$\frac{\partial g_1(x_1, x_2)}{\partial x_1}=\frac{1}{x_2}$

$\frac{\partial g_1(x_1, x_2)}{\partial x_2}=-\frac{x_1}{x_2^2}$

下面计算 $\frac{\partial x_1}{\partial Z_i}$

令 $x_1=g_2(x_3)=e^{x_3}$

令 $x_3=g_3(Z_i)=Z_i-max({{Z}_i})$

$\frac{\partial x_1}{\partial Z_i}=\frac{\partial g_2(x_3)}{\partial x_3}\frac{\partial g_3(Z_i)}{\partial Z_i}=e^{x_3}\cdot1=e^{Z_i-max({{Z}_i})}$

下面计算 $\frac{\partial x_2}{\partial Z_i}$

令 $x_2=g_4(x_4)=x_4+c_1$

其中 $c_1$ 为常数

令 $x_4=g_5(x_5)=e^{x_5}$

令 $x_5=g_6(Z_i)=Z_i-max({{Z}_i})$

$\frac{\partial x_2}{\partial Z_i}=\frac{\partial g_4(x_4)}{\partial x_4}\frac{\partial g_5(x_5)}{\partial x_5}\frac{\partial g_6(z_t)}{\partial Z_i}$

$\frac{\partial g_4(x_4)}{\partial x_4}=1$

$\frac{\partial g_5(x_5)}{\partial x_5}=e^{x_5}=e^{Z_i-max({{Z}_i})}$

$\frac{\partial g_6(z_t)}{\partial Z_i}=1$


$\frac{\partial x_2}{\partial Z_i}=e^{Z_i-max({{Z}_i})}$

故 $\frac{\partial softmax(Z_i)}{\partial Z_i}=\frac{1}{x_2}\cdot e^{Z_i-max({{Z}_i})}+(-\frac{x_1}{x_2^2})\cdot e^{Z_i-max({{Z}_i})}$

其中

$x_1=e^{Z_i-max({{Z}_i})}$

$x_2=sum$

故 $\frac{\partial softmax(Z_i)}{\partial Z_i}=\frac{e^{Z_i-max({{Z}_i})}}{sum}\cdot (1-\frac{e^{Z_i-max({{Z}_i})}}{sum})$

又因为 $softmax(Z_i)=\frac{e^{Z_i}}{sum}$

故 $\frac{\partial softmax(Z_i)}{\partial Z_i}=softmax(Z_i)\cdot (1-softmax(Z_i))$


当 $i \neq target$

令 $softmax(Z_{target})=g_1(x_1) = \frac{e^{Z_{target}-max({{Z}_i})}}{x_1} $

令 $x_1=g_2(x_2)=x_2+c_1$ 其中 $c_1$ 为常数

令 $x_2=g_3(x_3)=e^{x_3}$

令 $x_3=g_4(Z_i)=Z_i-max({{Z}_i})$

$\frac{\partial softmax(Z_{target})}{\partial Z_i}=\frac{\partial g_1(x_1)}{x_1}\frac{\partial g_2(x_2)}{\partial x_2}\frac{\partial g_3(x_3)}{\partial x_3}\frac{\partial g_4(Z_i)}{\partial Z_i}$

$\frac{\partial g_1(x_1)}{x_1}=-\frac{e^{Z_{target}-max({{Z}_i})}}{sum^2}$

$\frac{\partial g_2(x_2)}{\partial x_2}=1$

$\frac{\partial g_3(x_3)}{\partial x_3}=e^{x_3}=e^{Z_i-max({{Z}_i})}$

$\frac{\partial g_4(Z_i)}{\partial Z_i}=1$

故

```math
\frac{\partial softmax(Z_{target})}{\partial Z_i}=-\frac{e^{Z_{target}-max({{Z}_i})}}{sum}\cdot \frac{e^{Z_i-max({{Z}_i})}}{sum}=-softmax(Z_{target})\cdot softmax(Z_i)
```

最终整理

```math
\frac{\partial softmax(Z_{target})}{\partial Z_i}=\begin{cases}-softmax(Z_{target})\cdot softmax(Z_i), & \text{if } i \neq target \\
softmax(Z_i)\cdot (1-softmax(Z_i)), & \text{if } i = target
\end{cases}
```

## layernorm

参考 [https://zhuanlan.zhihu.com/p/634644501](https://zhuanlan.zhihu.com/p/634644501)

$Layernorm(x_i)=\gamma\frac{x_i-\mu}{\sigma}+\beta$

$\sigma=\sqrt{var+\epsilon}$

$var=\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2$

$\mu=\frac{1}{n}\sum_{i=1}^nx_i$

令 $\hat{x_i}=\frac{x_i-\mu}{\sigma}$

我们最关注 $\frac{\partial \hat{x_i}}{\partial x_j}$ ，乘以 $\gamma$ 和加上 $\beta$ 的部分交给node.h中的矩阵乘法加法的自动求导即可

只用链式法则

令 $x_i = g_1(x_1, x_2)$

令 $x_1 = x_i-\mu$

令 $x_2 = \sigma$

故

$\frac{\partial \hat{x_i}}{\partial x_j}=\frac{\partial g_1(x_1, x_2)}{\partial x_1}\frac{\partial xi-\mu}{\partial x_j}+\frac{\partial g_1(x_1, x_2)}{\partial x_2}\frac{\partial \sigma}{\partial x_j}$ (1)

$\frac{\partial g_1(x_1, x_2)}{\partial x_1}=\frac{1}{x_2}=\frac{1}{\sigma}$

$\frac{\partial xi-\mu}{\partial x_j}=\frac{\partial x_i}{\partial x_j}-\frac{\partial\mu}{\partial x_j}$

令 $\delta_{ij}=\frac{\partial x_i}{\partial x_j}$

故 $\delta_{ij}=\begin{cases}0, & \text{if } i \neq j \\
1, & \text{if } i = j
\end{cases}$
