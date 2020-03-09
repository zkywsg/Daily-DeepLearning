## Linux文件与目录管理

- 绝对路径:路径的写法，由根目录 / 写起，例如： /usr/share/doc 这个目录。
- 相对路径:不是由 / 写起，例如由 /usr/share/doc 要到 /usr/share/man 底下时，可以写成： cd ../man



## ls列出目录

- -a ：全部的文件，连同隐藏文件( 开头为 . 的文件) 一起列出来(常用)

- -d ：仅列出目录本身，而不是列出目录内的文件数据(常用)

- -l ：长数据串列出，包含文件的属性与权限等等数据；(常用)

- 将家目录下的所有文件列出来(含属性与隐藏档)

  - ```shell
    [root@www ~]# ls -al ~
    ```



## cd切换目录

- ```shell
   cd [相对路径或绝对路径]
  ```

- ```shell
  #使用 mkdir 命令创建 runoob 目录
  [root@www ~]# mkdir runoob
  
  #使用绝对路径切换到 runoob 目录
  [root@www ~]# cd /root/runoob/
  
  #使用相对路径切换到 runoob 目录
  [root@www ~]# cd ./runoob/
  
  # 表示回到自己的家目录，亦即是 /root 这个目录
  [root@www runoob]# cd ~
  
  # 表示去到目前的上一级目录，亦即是 /root 的上一级目录的意思；
  [root@www ~]# cd ..
  ```



## pwd 显示目前所在的目录

- ```shell
  [root@www ~]# pwd
  /root   <== 显示出目录啦～
  ```



## mkdir 创建新目录

- ```shell
  mkdir [-mp] 目录名称
  ```

  - -m ：配置文件的权限喔！直接配置，不需要看默认权限 (umask) 的脸色
  - -p ：帮助你直接将所需要的目录(包含上一级目录)递归创建起来

- ```shell
  [root@www ~]# cd /tmp
  [root@www tmp]# mkdir test    <==创建一名为 test 的新目录
  [root@www tmp]# mkdir test1/test2/test3/test4
  mkdir: cannot create directory `test1/test2/test3/test4': 
  No such file or directory       <== 没办法直接创建此目录啊！
  [root@www tmp]# mkdir -p test1/test2/test3/test4
  ```

- ```shell
  [root@www tmp]# mkdir -m 711 test2
  [root@www tmp]# ls -l
  drwxr-xr-x  3 root  root 4096 Jul 18 12:50 test
  drwxr-xr-x  3 root  root 4096 Jul 18 12:53 test1
  drwx--x--x  2 root  root 4096 Jul 18 12:54 test2
  ```



## rmdir删除空目录

- ```shell
   rmdir [-p] 目录名称
  ```

- -p:连同上一级『空的』目录也一起删除

- ```shell
  [root@www tmp]# rmdir runoob/
  ```

- ```shell
  [root@www tmp]# ls -l   <==看看有多少目录存在？
  drwxr-xr-x  3 root  root 4096 Jul 18 12:50 test
  drwxr-xr-x  3 root  root 4096 Jul 18 12:53 test1
  drwx--x--x  2 root  root 4096 Jul 18 12:54 test2
  [root@www tmp]# rmdir test   <==可直接删除掉，没问题
  [root@www tmp]# rmdir test1  <==因为尚有内容，所以无法删除！
  rmdir: `test1': Directory not empty
  [root@www tmp]# rmdir -p test1/test2/test3/test4
  [root@www tmp]# ls -l        <==您看看，底下的输出中test与test1不见了！
  drwx--x--x  2 root  root 4096 Jul 18 12:54 test2
  ```



## cp 复制文件或目录

- ```shell
  [root@www ~]# cp [-adfilprsu] 来源档(source) 目标档(destination)
  [root@www ~]# cp [options] source1 source2 source3 .... directory
  ```

- -a:相当於 -pdr 的意思

- -d:若来源档为连结档的属性(link file)，则复制连结档属性而非文件本身

- -f:为强制(force)的意思，若目标文件已经存在且无法开启，则移除后再尝试一次

- -i:若目标档(destination)已经存在时，在覆盖时会先询问动作的进行

- -p:连同文件的属性一起复制过去，而非使用默认属性

- -r:递归持续复制，用於目录的复制行为

- ```shell
  [root@www ~]# cp ~/.bashrc /tmp/bashrc
  [root@www ~]# cp -i ~/.bashrc /tmp/bashrc
  cp: overwrite `/tmp/bashrc'? n  <==n不覆盖，y为覆盖
  ```



## rm 移除文件或目录

- ```shell
  rm [-fir] 文件或目录
  ```

- -f:就是 force 的意思，忽略不存在的文件，不会出现警告信息

- -i:互动模式，在删除前会询问使用者是否动作

- -r:递归删除

- ```shell
  [root@www tmp]# rm -i bashrc
  rm: remove regular file `bashrc'? y
  ```



## mv 移动文件和目录/或修改名称

- ```shell
  [root@www ~]# mv [-fiu] source destination
  ```

- -f:force 强制的意思，如果目标文件已经存在，不会询问而直接覆盖

- -i:若目标文件 (destination) 已经存在时，就会询问是否覆盖

- -u:若目标文件已经存在，且 source 比较新，才会升级

- ```shell
  [root@www ~]# cd /tmp
  [root@www tmp]# cp ~/.bashrc bashrc
  [root@www tmp]# mkdir mvtest
  [root@www tmp]# mv bashrc mvtest
  ```



## cat 显示文件内容

- ```shell
  [root@www ~]# cat /etc/issue
  CentOS release 6.4 (Final)
  Kernel \r on an \m
  ```



## tac 文件内容从最后一行开始显示

- ```shell
  [root@www ~]# tac /etc/issue
  
  Kernel \r on an \m
  CentOS release 6.4 (Final)
  ```



## nl 显示行号

- ```shell
  [root@www ~]# nl /etc/issue
       1  CentOS release 6.4 (Final)
       2  Kernel \r on an \m
  ```



## more 一页一页翻动

- ```shell
  [root@www ~]# more /etc/man_db.config 
  #
  # Generated automatically from man.conf.in by the
  # configure script.
  #
  # man.conf from man-1.6d
  ....(中间省略)....
  --More--(28%)  <== 重点在这一行喔！你的光标也会在这里等待你的命令
  ```



