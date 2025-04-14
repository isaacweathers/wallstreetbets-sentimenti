"""
Topic Modeling Module

This module provides a TopicModeler class that implements Latent Dirichlet Allocation (LDA)
for topic modeling on text documents. It uses scikit-learn's implementation of LDA and
includes methods for preprocessing, fitting, and extracting topics from documents.

The TopicModeler class:
1. Preprocesses documents using a custom TextProcessor
2. Creates a document-term matrix using CountVectorizer
3. Fits an LDA model to the document-term matrix
4. Extracts topics and their keywords
5. Provides methods to get the most likely topic for each document
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from text_processor import TextProcessor

class TopicModeler:
    """
    A class for topic modeling using Latent Dirichlet Allocation (LDA).
    
    This class provides methods for preprocessing documents, fitting an LDA model,
    and extracting topics and their keywords. It uses scikit-learn's implementation
    of LDA and includes methods for getting the most likely topic for each document.
    
    Attributes:
        n_topics (int): Number of topics to extract
        max_iter (int): Maximum number of iterations for LDA
        learning_decay (float): Learning decay parameter for LDA
        random_state (int): Random state for reproducibility
        text_processor (TextProcessor): Text processor for preprocessing documents
        vectorizer (CountVectorizer): Vectorizer for creating document-term matrix
        lda_model (LatentDirichletAllocation): LDA model for topic modeling
        feature_names (list): List of feature names (words) from the vectorizer
        topics (list): List of topics, each containing (word, weight) tuples
    """
    
    def __init__(self, n_topics=5, max_iter=10, learning_decay=0.7, random_state=42):
        """
        Initialize the topic modeler with LDA parameters.
        
        Args:
            n_topics (int): Number of topics to extract
            max_iter (int): Maximum number of iterations for LDA
            learning_decay (float): Learning decay parameter for LDA
            random_state (int): Random state for reproducibility
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_decay = learning_decay
        self.random_state = random_state
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.vectorizer = CountVectorizer(
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            min_df=2,     # Ignore terms that appear in less than 2 documents
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with at least 2 letters
        )
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_decay=learning_decay,
            random_state=random_state,
            batch_size=128,
            n_jobs=-1  # Use all available cores
        )
        
        # Will be set after fitting
        self.feature_names = None
        self.topics = None
    
    def preprocess_documents(self, documents):
        """
        Preprocess documents for topic modeling.
        
        This method:
        1. Processes each document using TextProcessor
        2. Joins tokens back into strings
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: Preprocessed documents
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                print(f"Warning: Document {i} is not a string: {type(doc)}")
                continue
                
            # Process the text using our TextProcessor
            _, processed_tokens = self.text_processor.process_text(doc)
            
            if not processed_tokens:
                print(f"Warning: No tokens generated for document {i}")
                continue
                
            # Join tokens back into a string
            processed_doc = ' '.join(processed_tokens)
            if not processed_doc.strip():
                print(f"Warning: Empty document after processing document {i}")
                continue
                
            processed_docs.append(processed_doc)
        
        print(f"Processed {len(processed_docs)} documents out of {len(documents)} input documents")
        return processed_docs
    
    def fit(self, documents):
        """
        Fit the topic model to the documents.
        
        This method:
        1. Preprocesses the documents
        2. Creates a document-term matrix using the vectorizer
        3. Fits the LDA model to the document-term matrix
        4. Extracts topics and their keywords
        
        Args:
            documents (list): List of text documents
            
        Returns:
            self: The fitted model
        """
        if not documents:
            print("Warning: No documents provided for topic modeling")
            return self
            
        # Preprocess documents
        processed_docs = self.preprocess_documents(documents)
        if not processed_docs:
            print("Warning: No documents remained after preprocessing")
            return self
            
        print(f"Preprocessed first document: {processed_docs[0][:100]}...")
        
        # Create document-term matrix
        doc_term_matrix = self.vectorizer.fit_transform(processed_docs)
        print(f"Document-term matrix shape: {doc_term_matrix.shape}")
        
        # Get feature names (words)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Number of features (vocabulary size): {len(self.feature_names)}")
        
        # Fit LDA model
        self.lda_model.fit(doc_term_matrix)
        
        # Extract topics
        self.topics = self._extract_topics()
        print(f"Number of topics extracted: {len(self.topics)}")
        if self.topics:
            print("First topic keywords:", [word for word, _ in self.topics[0]])
        
        return self
    
    def _extract_topics(self, n_top_words=10):
        """
        Extract the top words for each topic.
        
        This method:
        1. Gets the components of the LDA model
        2. For each topic, gets the indices of the top words
        3. Gets the words and their weights
        4. Returns a list of topics, each containing (word, weight) tuples
        
        Args:
            n_top_words (int): Number of top words to extract per topic
            
        Returns:
            list: List of topics, each containing (word, weight) tuples
        """
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get indices of top words
            top_words_idx = topic.argsort()[:-n_top_words-1:-1]
            
            # Get the words and their weights
            top_words = []
            for idx in top_words_idx:
                word = self.feature_names[idx]
                weight = topic[idx] / topic.sum()  # Normalize weight
                top_words.append((word, weight))
            
            topics.append(top_words)
        
        return topics
    
    def transform(self, documents):
        """
        Transform documents into topic distributions.
        
        This method:
        1. Preprocesses the documents
        2. Creates a document-term matrix using the vectorizer
        3. Transforms the document-term matrix into topic distributions
        
        Args:
            documents (list): List of text documents
            
        Returns:
            numpy.ndarray: Topic distributions for each document
        """
        # Preprocess documents
        processed_docs = self.preprocess_documents(documents)
        
        # Create document-term matrix
        doc_term_matrix = self.vectorizer.transform(processed_docs)
        
        # Transform to topic distributions
        topic_distributions = self.lda_model.transform(doc_term_matrix)
        
        return topic_distributions
    
    def get_document_topics(self, documents):
        """
        Get the most likely topic for each document.
        
        This method:
        1. Transforms the documents into topic distributions
        2. Gets the most likely topic for each document
        3. Returns a list of (document, topic_idx, topic_probability) tuples
        
        Args:
            documents (list): List of text documents
            
        Returns:
            list: List of (document, topic_idx, topic_probability) tuples
        """
        # Get topic distributions
        topic_distributions = self.transform(documents)
        
        # Get most likely topic for each document
        document_topics = []
        
        for i, doc_dist in enumerate(topic_distributions):
            topic_idx = np.argmax(doc_dist)
            topic_prob = doc_dist[topic_idx]
            document_topics.append((documents[i], topic_idx, topic_prob))
        
        # Ensure we return the same number of topics as documents
        if len(document_topics) != len(documents):
            print(f"Warning: Number of document topics ({len(document_topics)}) does not match number of documents ({len(documents)})")
            # If we have fewer topics than documents, add placeholder topics
            while len(document_topics) < len(documents):
                document_topics.append((documents[len(document_topics)], 0, 0.0))
        
        return document_topics
    
    def get_topic_keywords(self, topic_idx):
        """
        Get the keywords for a specific topic.
        
        This method:
        1. Checks if the model is fitted and the topic index is valid
        2. Returns the keywords for the specified topic
        
        Args:
            topic_idx (int): Index of the topic
            
        Returns:
            list: List of (word, weight) tuples for the topic
        """
        if self.topics is None or topic_idx >= len(self.topics):
            return []
        
        return self.topics[topic_idx]
    
    def print_topics(self):
        """
        Print all topics and their keywords.
        
        This method:
        1. Checks if the model is fitted
        2. Prints each topic and its keywords
        """
        if self.topics is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        for i, topic in enumerate(self.topics):
            print(f"Topic {i+1}: {', '.join([word for word, _ in topic])}") 