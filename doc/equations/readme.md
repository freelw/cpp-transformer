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

$\frac{\partial g_1(x_1, x_2)}{\partial x_1}=\frac{1}{x_2}=\frac{1}{\sigma}$ (2)

$\frac{\partial xi-\mu}{\partial x_j}=\frac{\partial x_i}{\partial x_j}-\frac{\partial\mu}{\partial x_j}$

令 $\delta_{ij}=\frac{\partial x_i}{\partial x_j}$

故 
```math
\delta_{ij}=\begin{cases}0, & \text{if } i \neq j \\
1, & \text{if } i = j
\end{cases}
```

再计算 $\frac{\partial\mu}{\partial x_j}=\frac{\partial \frac{1}{n}\sum_{i=1}^nx_i}{\partial x_j}=\frac{1}{n}$

故 $\frac{\partial xi-\mu}{\partial x_j}=\frac{\partial x_i}{\partial x_j}-\frac{\partial\mu}{\partial x_j}=\delta_{ij}-\frac{1}{n}$ (3)

$\frac{\partial g_1(x_1, x_2)}{\partial x_2}=-\frac{x_1}{x_2^2}=\frac{x_i-\mu}{\sigma^2}$ (4)

$\frac{\partial \sigma}{\partial x_j}=\frac{\partial sigma}{\partial var}\frac{\partial var}{\partial x_j}$ (5)

$\frac{\partial var}{\partial x_j}=\frac{\partial \frac{1}{n}\sum_{k=1}^n(x_k-\mu)^2}{\partial x_j}=\frac{1}{n}\sum_{k=1}^n2(x_k-\mu)\frac{\partial x_k-\mu}{\partial x_j}$

深入分析 $\frac{1}{n}\sum_{k=1}^n2(x_k-\mu)\frac{\partial x_k-\mu}{\partial x_j}$

```math
\frac{\partial x_k-\mu}{\partial x_j}=\begin{cases}-\frac{1}{n}, & \text{if } i \neq j \\
1-\frac{1}{n}, & \text{if } i = j
\end{cases}
```

所以展开 $\sum_{k=1}^n2(x_k-\mu)\frac{\partial x_k-\mu}{\partial x_j}=\sum_{\substack{i = 1 \\ i \neq k}}^n-\frac{1}{n}\cdot2\cdot(x_k-\mu)+2\cdot(x_j-\mu)(1-\frac{1}{n})=2\cdot(-\frac{1}{n}\sum_{\substack{i = 1 \\ i \neq k}}^nx_k+\frac{1}{n}(n-1)\mu+x_j-\frac{1}{n}\cdot x_j-\mu+\frac{1}{n}\cdot\mu)$

观察第一项和第四项的和 $-\frac{1}{n}\sum_{\substack{i = 1 \\ i \neq k}}^nx_k-\frac{1}{n}\cdot x_j=-\mu$

观察第二项和第六项的和 $\frac{1}{n}(n-1)\mu+\frac{1}{n}\cdot\mu=\mu$

上面两个和相加消掉了，只剩下第三项和第五项

故 $\frac{\partial var}{\partial x_j}=\frac{2}{n}(x_j-\mu)$

又因为 $\frac{\partial \sigma}{\partial var}=\frac{\partial \sqrt(var+\epsilon)}{\partial var}=\frac{1}{2\sqrt(var+\epsilon)}=\frac{1}{2\sigma}$

故 (5)= $\frac{\partial \sigma}{\partial x_j}=\frac{\partial sigma}{\partial var}\frac{\partial var}{\partial x_j}=\frac{1}{2\sigma}\cdot\frac{2}{n}(x_j-\mu)=\frac{1}{n\sigma}(x_j-\mu)$

将(2)(3)(4)(5) 带回 (1)

$\frac{\partial \hat{x_i}}{\partial x_j}=\frac{1}{\sigma}(\delta_{ij}-\frac{1}{n})+\frac{x_i-\mu}{\sigma^2}\cdot\frac{1}{n\sigma}(x_j-\mu)$

观察这里的第二项 $\frac{x_i-\mu}{\sigma^2}\cdot\frac{1}{n\sigma}(x_j-\mu)=\frac{1}{n\sigma}\cdot\frac{x_i-\mu}{\sigma}\cdot\frac{x_j-\mu}{\sigma}=\frac{1}{n\sigma}\cdot\hat{x_i}\cdot\hat{x_j}$

故

```math
\frac{\partial \hat{x_i}}{\partial x_j}=\frac{1}{\sigma}(\delta_{ij}-\frac{1}{n})+\frac{1}{n\sigma}\cdot\hat{x_i}\cdot\hat{x_j}=\frac{\delta_{ij}}{\sigma}-\frac{1}{n\sigma}-\frac{1}{\sigma}\hat{x_i}\hat{x_j}
```

其中

```math
\delta_{ij}=\begin{cases}0, & \text{if } i \neq j \\
1, & \text{if } i = j
\end{cases}
```

具体实现参见[代码](https://github.com/freelw/cpp-transformer/blob/df687a55ff57fcb8a2075283d64a5c06a41e5f5a/autograd/node.h#L488)