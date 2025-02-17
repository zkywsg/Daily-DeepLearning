### HDFS介绍

- Hadoop Distribute File System /Hadoop分布式文件系统
- 作为最底层的分布式存储服务而存在

![](https://imgkr.cn-bj.ufileos.com/c0486546-6b1c-42d6-b833-18ab97590a63.png)

- 备份存储
- ![](https://imgkr.cn-bj.ufileos.com/abb2b658-395a-4ae1-ab1d-91bc46aeccec.png)



### HDFS设计目标

- 硬件故障是常态，故障检测和自动快速回复是HDFS的核心架构目标。
- HDFS上的应用与一般的应用不同，主要是以流式读取。相交于数据访问的反应时间，更注重数据访问的高吞吐量。
- 典型的HDFS文件大小是GB到TB级别。
- HDFS应用对文件的要求是write-one-read-many访问模型。一个文件一旦创建、写入、关闭之后就不需要修改了。这个假设简化了数据一致性问题，使高吞吐量的数据访问成为可能。
- 移动计算的代价比移动数据代价低。一个应用请求的计算，离它操作的数据越近就越高效，在数据达到海量级别的时候也是。
- 可移植性





### HDFS重要特性

- 是一个文件系统，用于存储文件，通过统一的命名空间目录树来定位文件
- 分布式的，集群中的服务器各自有角色



- master/slave架构
  - 一般一个HDFS集群是一个Namenode和一定数目的Datanode组成。
  - NameNode是HDFS集群主节点，DataNode是HDFS集群从结点，两种角色共同协调完成分布式的文件存储服务。
- 分块存储
  - HDFS中的文件在物理上是分块存储的，块的大小可以通过配置参数来规定 。
- 名字空间
  - NameNode负责维护文件系统的名字空间，任何对文件系统名字空间或属性的修改都将被Namenode记录下来。
  - HDFS会给客户端提供一个统一的抽象目录树
    - hdfs://namenode:port/dir-a/dir-b/file.data



### NameNode元数据管理

- 把目录结构和文件分块位置信息叫作元数据。
- NameNode负责维护整个hdfs文件系统的目录树结构和每个文件对应的block块信息。



### DataNode数据存储

- 文件的各个block的具体存储管理由datanode节点承担。每个block都可以在多个datanode上。
- Datanode需要定时向namenode汇报自己持有的block信息。
- 存储多个副本

### 副本机制

- 为了容错，文件的所有block都有副本。
- 应用程序可以指定某个文件的副本数目。
- 副本系数可以在文件创建的时候指定，也可以在之后改变。



### 一次写入，多次读出

- HDFS是设计成适应一次写入，多次读出的场景，不支持文件的修改。 
- HDFS适合用来组大数据分析的底层存储服务。