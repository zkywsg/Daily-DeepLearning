# from urllib import request,error
# try:
#     response = request.urlopen('http://cuiqingcai.com/index.htm')
# except error.URLError as e:
#     print(e.reason)
#
# from urllib import request,error
#
# try:
#     response = request.urlopen('http://cuiqingcai.com/index.htm')
# except error.HTTPError as e:
#     print(e.reason,e.code,e.headers,sep='\n')

# from urllib import request,error
#
# try:
#     response = request.urlopen('http://cuiqingcai.com/index.htm')
# except error.HTTPError as e:
#     print(e.reason,e.code,e.headers,sep='\n')
# except error.URLError as e:
#     print(e.reason)
# else:
#     print('Request Successfully')

import socket
import urllib.request
import urllib.error

try:
    response = urllib.request.urlopen('https://www.zhihu.com',timeout=0.01)
except urllib.error.URLError as e:
    print(type(e.reason))
    if isinstance(e.reason,socket.timeout):
        print('TIME OUT')