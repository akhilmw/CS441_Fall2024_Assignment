# Author: Akhil S Nair
# Email: anair56@uic.edu

## Introduction
This project implements a distributed system to generate word embeddings and compute semantic similarities between words using Hadoop MapReduce and ND4J. The main goal is to process large datasets in a distributed cloud environment (AWS EMR) and derive meaningful insights into word relationships using embeddings and cosine similarity. The project was designed as part of the CS 441: Distributed Objects in Cloud Computing course, taught by Prof. Dr. Mark Grechanik at the University of Illinois Chicago.

**Video Link:** [https://youtu.be/Oc87pyydd-0?si=_ho_Z8h6a27IhDTO]  
(The video explains the deployment of the Hadoop application in the AWS EMR Cluster and the project structure.)

## Environment
- **OS:** Mac
- **IDE:** IntelliJ IDEA 2023.3.8 (Ultimate Edition)
- **Scala Version:** 2.13.12
- **SBT Version:** 1.10.1
- **Hadoop Version:** 3.3.6

## Components

1. **ShardingUtil**: Divides large text datasets into smaller shards for distributed processing.
2. **WordCountBPEJob**: A MapReduce job to count word frequencies across the text corpus.
3. **OrderedWordMR**: A MapReduce job that maintains the order of words and generates the tokens.
4. **EmbeddingsGenerator**: Combines tokenized data and prepares it for embedding generation.
5. **SemanticRepresentation**: A MapReduce task that computes semantic similarities between words based on their embeddings.

## Prerequisites
Before starting the project, ensure you have the following tools and accounts set up:
- **Hadoop:** Install and configure Hadoop on your local machine or cluster.
- **AWS Account:** Create an AWS account and familiarize yourself with AWS Elastic MapReduce (EMR).
- **Java:** Ensure that Java is installed and properly configured.
- **Git and GitHub:** Use Git for version control and host your project repository on GitHub.
- **IDE:** Choose an Integrated Development Environment (IDE) for coding and development.

## Commands to Clean, Compile, and Run Locally


### Running the Test File
Test files can be found under the directory `src/test`:
```bash
sbt clean compile test
```

### To clean, compile, and run locally:
```bash
sbt clean compile run
```

## Conclusion
This project successfully demonstrates the application of Hadoop MapReduce and ND4J for distributed word embedding generation and semantic similarity computation. By leveraging AWS EMR, we've shown how large datasets can be processed efficiently in a cloud-based environment.
The integration of modern NLP tools like JTokkit for tokenization and Deeplearning4j for embedding generation showcases the power of combining cutting-edge libraries with distributed systems for large-scale data processing. This project serves as a practical example of how cloud computing technologies and distributed systems can be effectively applied to solve complex natural language processing tasks.

## Acknowledgments
I would like to thank Prof. Dr. Mark Grechanik for his guidance and instruction throughout the CS 441 course, which greatly contributed to the success of this project.
