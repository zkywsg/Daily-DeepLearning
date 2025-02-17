### HDFS基本操作

- hadoop fs <args> 
- hadoop fs -ls hdfs://node-1:9000  (查看)
- hadoop fs -ls file:///root (linux下的查看 本地)
- hdfs dfs -ls / (也是查看 旧版本)

### 常见命令参考

- -ls 查看指定路径的当前目录结构
  - hadoop fs -ls /hello
  - hadoop fs -ls -h /hello  (显示文件大小)
- -mv 移动
- -cp 复制
- -rm 删除文件
- -put 上传文件
- -cat 查看文件内容

- 