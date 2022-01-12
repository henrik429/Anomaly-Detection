# Anomaly Detection
As the name said, anomaly detection is the identification of anomalies in data. 
Instances of data are considered as anomalies, if they differ significantly from the
majority of the data and therefore, raise suspicions. 
Thus, they are also called outliers, noise or novelties.
So, anomaly detection is the identification of rare items, events or observations
[[1](https://avinetworks.com/glossary/anomaly-detection/),
[2](https://en.wikipedia.org/wiki/Anomaly_detection),
[3](https://nix-united.com/blog/machine-learning-for-anomaly-detection-in-depth-overview/)]. 

In context of Cybersecurity, applications of anomaly detection can vary widely. 
It can be used to detect 
unauthorized access attempts or suspicious activity such as unusual types of 
requests. It is applicable for 
fraud or intrusion detection and can also prevent sensitive data leaks 
[[2](https://en.wikipedia.org/wiki/Anomaly_detection), 
[3](https://nix-united.com/blog/machine-learning-for-anomaly-detection-in-depth-overview/)].

There are three main classes of anomaly detection techniques: unsupervised, semi-supervised, and supervised. 
This repository refers to an unsupervised approach for anomaly detection. One of the most important
assumptions is here that the dataset used for the learning purpose contains mainly non-anomalous data and
only a small partition of the dataset is malicious and abnormal. Thus, unsupervised 
anomaly detection algorithms deem collections of frequent, similar instances to be
normal and identify infrequent data groups as malicious
[[1](https://avinetworks.com/glossary/anomaly-detection/)].

## Adversarially Learned Anomaly Detection[*](https://arxiv.org/pdf/1812.02288.pdf)
This repository is based on the Paper [Adversarially Learned Anomaly Detection](https://arxiv.org/pdf/1812.02288.pdf). 
In this work, they propoesed a new anomaly detection method, named Adversarially Learned Anomaly Detection (ALAD)
that is predicated on bi-directional GANs. ALAD outputs adversarially learned features which are then
used for the anomaly detection task to determine if a data sample is anomalous by utilizing reconstructions errors. 

