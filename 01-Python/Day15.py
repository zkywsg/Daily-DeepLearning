# coding:utf-8

# Author: @zkywsg

# data = ['zkywsg',23,(1997,1,1)]
# name,years,birth = data
# print name,years,birth

# string = 'sb'
# a,b = string
# print a,b

info = ('zkywsg',23,'phone1','phone2')
name,years,*numbers = info
print(name,years,numbers)