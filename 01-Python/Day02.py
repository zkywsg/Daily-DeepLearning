# %% markdown
# ### 1. 列表
# - 1.1 访问列表中的值
# - 1.2 修改/删除/添加元素
# - 1.3 列表中的常用操作
# - 1.4 列表中的常用函数
# %%
# 1.1 访问列表中的值 索引从0开始
name = ['Tim','Kris','Sam']

# 1.1.1 访问列表中的第一个值
print('访问列表中的第一个值:',name[0])
print('----------')

# 1.1.2 访问第二个以后所有值
print('访问列表中的第一个值:',name[1:])
print('----------')

# 1.1.3 访问所有值
print('访问所有值:',name)
print('----------')

# 1.1.4 访问倒数第一个
print('访问倒数第一个:',name[-1])
# %%
# 1.2 修改/删除/添加元素

# 1.2.1 修改元素
name[2] = 'sb'
print('修改第3个位置的元素后:',name[2])
print('----------')

# 1.2.2 删除元素
# 删除任意位置的元素 del()函数

print('删除元素前：',name)
del name[1]
print('删除第2个元素后:',name)
print('----------')

# 删除最后一个元素 pop()函数
print('删除元素前：',name)
name.pop()
print('删除最后一个元素后:',name)
print('----------')

# 1.2.3 添加元素
# 在任意位置添加元素 insert()函数
print('添加元素前:',name)
name.insert(1,'Kris')
print('在第1个位置添加元素后:',name)
print('----------')

# 在最后一个位置添加元素 append()函数
print('添加元素前:',name)
name.append('Sam')
print('添加在最后一个位置后:',name)
# %%
# 1.3 列表中的常用操作
list1 = [1,2,3]
list2 = [4,5,6]
list3 = [1]

# 列表的+操作
print("两个列表相加后:",list1+list2)

# 列表的*操作
print('列表所有元素x3:',list3*3)

# 列表的in， 判断元素是否在列表中
print('2是不是list1的元素:',2 in list1)

# for ... in 列表的遍历
for x in list1:
    print(x)
# %%
# 1.4 列表中常用的函数

# 获得列表的长度 len()函数
print('输出函数长度:',len(list1))
print('----------')

# 获得列表中的最大值 max()函数
print('列表中的最大值:',max(list1))
print('----------')

# 获得列表中的最小值 min()函数
print('列表中的最小值:',min(list1))

# 永久排序方法 sort()
list4 = [2,3,1,5]
list4.sort()
print('排序后的list4:',list4)
print('----------')

# 暂时排序方法 sorted()
list5 = [3,5,2,6,1]
print('暂时排序的结果:',sorted(list5))
print('排序后原列表:',list5)

# reverse=True参数 逆序排序
print('暂时逆序排序的结果:',sorted(list5,reverse=True))
# %% markdown
# ### 2. 元组
# - 2.1 定义元组
# - 2.2 访问/修改/删除元组
# %%
# 2.1 定义元组
t = (1,2)
print(t)

# 空元组
p = ()
print(p)
# %%
# 元组是不可修改的
t[0] = 2
# %%
# 2.2 访问/修改/删除元组

# 访问
print('访问第一个元素:',t[0])

# 修改是非法的 但是可以创建新的元组
t1 = (1,2,3)
t2 = (4,5)
print('拼接两个元组:',t1+t2)
print('---------')

# 删除
print('t3删除前:',t3)
del t3
# 删除后 就不存在了
print('t3删除后：',t3)
