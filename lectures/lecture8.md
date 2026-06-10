 PDF To Markdown Converter
Debug View
Result View
DS4CS Lecture 8 - Malware Detection in Linux Cloud Env.
Dept. of Industrial Engineering and Management, Ben-Gurion University
Trusted Detection of Unknown Executable
Malware in Linux-based Cloud
Environments Using Data Science Methods
DS 4 CS Course
Part of the Slides Are based
on lectures of:
Mr. Tomer Panker
Mr. Tom Landam
DS 4 C Lecture - 8
DR. NIR NISSIM
nirni@bgu.ac.il

Outline

Linux OS and Cloud Environments
Linux Malware Behavior
Related Work
Knowledge-Based Feature Extraction
Experimental Design & Results
What is Linux?

Linux is a free and open-source operating system (OS)
Linux empowers a variety of platforms and devices, including
personal computers, servers, and embedded systems
Android OS is based on the Linux kernel, which provides it with its
foundation, including memory management, and file system access
According to Verified Market Research (VMR) the cloud security market
is about to grow in about 26 % annually from 6.76 billion dollar in 2019
to 37.69 billion dollar in 2027
1
The Web and Email Security is expected to account for the largest
market share, due to more companies accepting the cloud-based services
The major factors driving the cloud security market include:
the increasing number of sophisticated cyber-attacks on cloud
computing systems,
growing need for compliance with various upcoming regulations
1 https://www.verifiedmarketresearch.com/product/global-cloud-security-market-size-and-forecast-to- 2025 /
Introduction – Cloud Market and Forecast
8
Linux is a family of operating systems (Debian, ubuntoetc.)
Linux-based clouds are popular cloud environments because Linux is a:
free
open-source,
high performing OS
suitable for multiple computational platforms
Linux is the dominant OS for cloud environments:
GCP – 90 %
MS Azure – 86 %
AWS – 82 %
Trusted Detection of Malware in VMs’ Cloud Environments - intro
Nowadays, many organizations implement privatecloud-computing environments
Centralized and easier maintenance
Better computational resource utilization
Better flexibility in computational resources allocations for different times and needs
Save of money
Many organizations also use publiccloud computing services (e.g. Amazon’s AWS)
Better computing performance
Almost no maintenance is needed by the customer organization
Virtualizationtechnology is at the coreof cloud computing.
Virtualization platforms: Vsphere(Vmware-Dell) , HyperV(Microsoft), VirtualBox(Linux) and more
18
Cloud Computing Environments
From Physical to Virtual Servers

Organizations use variety of application servers for their regular daily work
Web Server: IIS Server (Win) \ Apache Server (Linux)
Email Server: (MS Exchange)
Moodle Server , Course Registration Server
DNS Server: (bind 9 etc..)
FTP Server And more....
In the past, organizations used to purchase many physical servers (powerful computer) to
run their application servers (software that responds to variety of client requests)
Nowadays, Virtual servers (Virtual Machine or VMs), are commonly used to run
the application servers and provide services to the entire organization.
One powerful physical server can run multiple Virtual servers and their
applications
For performance and utilization reasons, one virtual server is dedicated for one
application server.
The Hypervisor – Manages the Virtual Machines
A component
(software / firmware / hardware)
which manages the VMs on the server
Canseeall
Cantakesnapshots
Invisible
Hypervisor

Hypervisor
WEB
Course
Registration
Exams &
Appeals
EMAIL
vCenter
Server
Cloud Admin
DNS
Moodle
From Physical to Virtual Servers
Clients
(Students \ Segel)
Clients
(Students \Segel)
Clients
(Students \Segel)

Linux Malware
Behavior
Taxonomy of potential behaviour indicative of Linux malware

Virtual servers are an attractive targeted for cyber attacks
Run critical organizational applications & Servers → RAT
DNS Server, Email Server, SQL Server ,IIS Server (Also internal ones: Moodle, Exams etc.)
Connected to the Storage Area Networks (SAN) → Ransomware
Comprised of powerfulcomputational resources → Crypto-Jacking
In many cases, virtual server had been compromised for long time, yet the
administrator is not aware of it.
In other cases, the attacks are detected after a severe damage has already
been done
Malware and Security of Cloud Environments
Motivation
Virtualization - Type

Virtualization - Type

Existing detection mechanisms are limited in their capabilities

Not Trusted since are installed over the virtual server itself, thusMalware can detect them, and:
Malware can manipulatethem
Malware can evade them and suspend its malicious activity (in dynamic analysis)
Malware can also turn them off
Incapableof detecting new unknown malware (Do not have Generalization Capability)
Most of them deterministic and based on signaturesor specific malware behavior patterns
Can be evaded by Encryption \Obfuscation \Packers. (in static analysis like AVs)
A better solution is needed, with the essential properties:

Trusted: Malware is not aware and can’t interfere the detection mechanism
Resilient to evasion techniques both static and dynamic
Able to detect new unknown malware (Generalization Capability)
Malware and Security of Cloud Environments – the Gap
41
Related Work
Trusted Detection of Malware in VMs ’ Cloud Environments – Gaps ( 1 )

(^)

No Trusted Malware Detection
Solution for Linux VMs
Trusted=
A mechanism that cannot be
interfered or manipulated by the
inspected machine and the
running applications
The inspected machine and
running applications are not
aware of the detection mechanism
No maximal utilization of VM’s
Volatile Memory Indications for
Malware Detection
- None of the prior studies
provided a trusted solution that
incorporates a wide variety of
feature sets to detect malicious
activity.
- Most of them used only one
source of features
(^)
Trusted Detection of Malware in VMs ’ Cloud Environments – Gaps ( 2 )

Methods
Trusted Acquisition of Volatile Memory Dump

Utilizing the separation between the guest and host in virtualization
We executed a server VM in an Oracle Virtualbox and acquired the Server’svolatile memory
by querying the hypervisor while executing malicious or benign programs in the server
The process is being conducted while the VM is momentarily frozen
An Overview of our Proposed Detection Framework
( 1 )
Trusted
Acquisition of
Volatile
memory Dump
( 2 )
Methodologies
for Feature
Extraction from
the Volatile
memory Dump
( 3 )
ML based Model
Training
( 4 )
Detection and
Evaluation
Advantages:
Allows overcoming malware’s ability to detect the security mechanism and evade detection
Resilient to static analysis evasion techniques (Packing, Obsucation, Dynamic code load)
It hasa generalization capability in detecting unknown malware as the framework is ML-based
Capable of detecting malicious activity in the VM, also when there is no application (fileless)
2
Hypervisor
1 GB 1 GB^1 GB
1 GB

Trusted: Snapshots of the VM are taken (RAM and HDD), using external client
Snapshots are taken every X seconds during server’s runtime and applications execution
Malware is not aware and can’t interfere the detection mechanism
Resilient to many evasion techniques:
The snapshot contains clear data of the RAM (not: encrypted \obfuscated \packed)
The malware is not aware that it is being monitored (does not delay its malicious activity in purpose)
Snapshots represent the state of the virtual machine (e.g. processes, libraries loaded, network etc.)
Capable of detecting new unknown malware that has continues behavior
Snapshots of volatile memory (RAM) represent the behavior of the server and all its applications
Snapshots are statically analyzed and the extracted features are leverage by Machine Learning algorithms
Characteristics of Our Proposed Solution
Might be start of
malicious activity
Δt Δt Δt Δt
Motivation – Supervised detection over time

We used two widely used application servers in organizations: HTTP and DNS servers
These servers are different inthe way they act
We were able to evaluate in different environments and conduct cross-server malware detection.
Two Virtual Servers were Examined
Benign sample collection includes 56 samples
54 benign applications and
two additional samples representing the states of the VM itself
All samples were verified to run in cloud env.
The benign applications reflect and simulates the usual activities performed in a real-world
virtual server
- Such as: VM's management
- Wireshark , Bmon
- Zip_Files
- Ping
- Installations (Python , Java, ruby)
- defrag
- Administrator operations
- opening document (xlsx, PDFs)
- searching in web
Data Collection – Benign ( 56 )
52
Malwaresamples collection includes 50 samples from different categories
Including ransomware, crypto-jacking, Trojans, viruses, botnets, and APT
Also: 3 more malicious samples were file-less attacks against the server
All samples were verified to run in cloud env.
Data Collection – Malware ( 53 )
In total we had 109 applications ( 56 benign + 53 malicious)
Every application ( 109 ) was injected and executed within the operating virtual server
Then, every 10 sec the volatile memory dump of the VM was taken - till 100 dumps
This process resulted a total of 21,800 volatile memory snapshots:
taken from two widely-used virtual servers
during the execution of a representative set of Linux applications.
Each snapshot carries different behavioral of the VM in different time point
Each dump was in size of 1.1GB
Our entire collection was in size of 24 TB
Volatile Memory Snapshot Collection
Our Proposed Solution – a Bird’s-Eye View
( 1 )
VMs Infrastructure
&
Client Simulation
( 2 )
Apps Injection
&
Snapshotting
( 3 )
Feature Engineering
Feature Representation Learning KBS Feature Extraction
( 4 )
Models Training
&
Ensemble
&
Detection
56
& MinHash &

Knowledge Based Features
A comprehensive feature set consisting of 171 features based on various
behavioral traces from different parts of the volatile memory
- 19 features were related to System-Calls information (that does not have Volatility plugin)
- 152 features were defined on raw data extracted by Volatility
The Extraction Process
Features Distributions

(^)
Malicious behavior Source

Proposed Solution - Features set (some Examples)
Behaviors Feature name Description Data source
Generic Behavior tmp_inode amount of /var/tmpinodes files
Generic Behavior no_open_file_percent percent of process without open files files
Generic Behavior anon_inode_amount amount of anon inode files
Generic Behavior mount_amount total amount of files mounted files
Generic Behavior avg_r-x_flag_per_proc average r-x flags per process files
Generic Behavior median_____flag_per_proc median ---flags per process files
Generic Behavior run_map_amount rum in proc maps files
Generic Behavior proc_inode amount of /proc inodes kernel
Information gathering sys_amount amount of files from /sysfolder that are in memory kernel
Information gathering active_ARP_dentry amount of /proc/net/arp that are in the memory
network
Generic Behavior localhost_percent percent of interfaces for localhost network
Information gathering established_conn_amount amount of established connections network
Generic Behavior sudo_arguments_amount amount of process with sudocommand process
Generic Behavior parameter_amount amount of parameters in psaux process
Deception empty_process_name amount of processes with empty names process
Process Interaction
procces_with_no_threads_percent
percent of process without threads(not include one main
thread) process
Generic Behavior sys_restart_syscall sys_restart_syscall system call

KB Feature Extraction

The features are aimed explicitly
at the malware detection task
A smaller amount of features with
high discriminability
Explainability
Does not utilize all the data in the dump
Requires a human expert knowledge
Limited to the cyber security’s expert knowledge
Expensive
Long processing \engineering time
Might need to be updated in the long run
Advantages Disadvantages
Experimental Design
&
Results

Training the
ML based
Detector

ï
Logistic regression
Support vector machines (SVM)
K-nearest neighbors (KNN)
Random forest
Artificial neural networks (ANNs)
Deep neural networks (DNN)
Knowledge-based dataset:
Similarity classifier –based on the Jaccarddistance
ï
Random forest
MinHash dataset:
Is it possible to detect unknownLinux malware in cloud environments in a trusted manner?
ResearchQuestion - 1 (unknown malware)
DNS serverbest results
TPR = 0.937, FPR = 0.008 , AUC = 0.964
DNNclassifier
HTTPserverbest results
TPR = 0.997, FPR = 0 , AUC = 0.999
DNN classifier
(^)

Core capability to detect new unknown malware – known malware can be detected by basic antivirus
74
Human Expert – Knowledge Based Features + ML
Is it possible to detect unknown Linux malwarecategoriesin cloud environments in a trusted manner?
An ability to deal with new malware that presentsa new type of malicious behavior
ResearchQuestion – 2 (entire unknown category)
DNSserverbest results
TPR = 0.9, FPR = 0.098 , AUC = 0.939
DNN classifier
HTTP serverbest results
TPR = 0. 918 , FPR = 0 , AUC = 0. 99
RF classifier
(^)
(^)
(^)
Human Expert – Knowledge Based Features + ML

Is it possible to accurately categorize Linux malware in cloud environments in a trusted manner?
Critical capability for response teams(identification of the specific attack)
Research Question – 3 (multiclass \categorization)
DNSserverbest results
ACC = 0.981
RF classifier
HTTP serverbest results
ACC = 0.98 6
DNN classifier
(^)
(^)
Human Expert – Knowledge Based Features + ML

Can a detection model trained on volatile memory dumps from one serverbe used to
detect unknown Linux malware in a trusted manner on a different server?
Critical capability for deployment in real-world companies that have several servers
Research Question – 4 (Cross Server Detection)
DNS-> HTTPserverbest results
TPR = 0.97, FPR = 0.087 , AUC = 0.942
RF classifier
HTTP -> DNSserverbest results
TPR = 0.724, FPR = 0.213 , AUC = 0.755
RF classifier
(^)
(^)
(^)
(^)
(^)
(^)
(^)
(^)
(^)
82
Human Expert – Knowledge Based Features + ML

Is it possible to detect unknown fileless attacks in cloud environments in a trusted manner?
A detection capability to deal with a newdifferent type of malicious activities against the servers
Research Question – 5 (New Modus Operandi)
DNSserverbest results
TPR = 1 , FPR = 1 , AUC = 1
HTTP serverbest results
TPR = 1 , FPR = 0 , AUC = 1
(^)
(^)
Human Expert – Knowledge Based Features + ML 85

This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports