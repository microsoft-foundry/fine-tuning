"""
Synthetic Data Analysis Tool for Fine-Tuning Data

Analyzes the JSONL fine-tuning data files to provide statistics, patterns,
and insights about synthetic conversations with visual charts.
Includes AI-powered conversation classification and clustering analysis.
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10


class SyntheticDataAnalyzer:
    """Analyze synthetic fine-tuning conversation data."""
    
    def __init__(self, train_path="data/datagen-ToolUse-FineTuneSupervised-train.jsonl",
                 valid_path="data/datagen-ToolUse-FineTuneSupervised-valid.jsonl",
                 create_charts=True):
        """Initialize analyzer with file paths."""
        self.train_path = train_path
        self.valid_path = valid_path
        self.create_charts = create_charts
        self.charts_created = []
        
        self.train_data = []
        self.valid_data = []
        self.all_data = []
        
        # Azure OpenAI configuration for AI classification
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        self.load_data()
    
    def load_data(self):
        """Load JSONL data files."""
        print(f"{Fore.CYAN}Loading synthetic data files...{Style.RESET_ALL}\n")
        
        # Load training data
        if os.path.exists(self.train_path):
            with open(self.train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.train_data.append(json.loads(line.strip()))
            print(f"{Fore.GREEN}✓ Loaded {len(self.train_data)} training conversations from {self.train_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Training file not found: {self.train_path}{Style.RESET_ALL}")
        
        # Load validation data
        if os.path.exists(self.valid_path):
            with open(self.valid_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.valid_data.append(json.loads(line.strip()))
            print(f"{Fore.GREEN}✓ Loaded {len(self.valid_data)} validation conversations from {self.valid_path}{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.RED}✗ Validation file not found: {self.valid_path}{Style.RESET_ALL}\n")
        
        self.all_data = self.train_data + self.valid_data
    
    def analyze_basic_statistics(self):
        """Analyze basic conversation statistics."""
        print(f"{Fore.CYAN}{'='*80}")
        print("📊 BASIC STATISTICS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        print(f"Dataset Split:")
        print(f"  Training conversations: {Fore.YELLOW}{len(self.train_data)}{Style.RESET_ALL}")
        print(f"  Validation conversations: {Fore.YELLOW}{len(self.valid_data)}{Style.RESET_ALL}")
        print(f"  Total conversations: {Fore.YELLOW}{len(self.all_data)}{Style.RESET_ALL}\n")
        
        # Analyze conversation lengths
        message_counts = []
        for conv in self.all_data:
            messages = conv.get('messages', [])
            # Count only non-system messages
            non_system = [m for m in messages if m.get('role') != 'system']
            message_counts.append(len(non_system))
        
        print(f"Conversation Length (messages):")
        print(f"  Min: {min(message_counts)}")
        print(f"  Max: {max(message_counts)}")
        print(f"  Average: {sum(message_counts)/len(message_counts):.2f}")
        print(f"  Median: {sorted(message_counts)[len(message_counts)//2]}\n")
        
        # Analyze message types
        role_counter = Counter()
        for conv in self.all_data:
            for msg in conv.get('messages', []):
                role_counter[msg.get('role', 'unknown')] += 1
        
        print(f"Message Distribution by Role:")
        total_messages = sum(role_counter.values())
        for role, count in role_counter.most_common():
            percentage = (count / total_messages) * 100
            print(f"  {role:12s}: {count:5d} ({percentage:5.2f}%)")
        print()
        
        return {
            'train_count': len(self.train_data),
            'valid_count': len(self.valid_data),
            'avg_length': sum(message_counts)/len(message_counts),
            'role_distribution': dict(role_counter)
        }
    
    def analyze_tool_usage(self):
        """Analyze tool call patterns."""
        print(f"{Fore.CYAN}{'='*80}")
        print("🔧 TOOL USAGE ANALYSIS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        # Extract tool definitions from first conversation
        tool_names = set()
        if self.all_data:
            tools = self.all_data[0].get('tools', [])
            for tool in tools:
                func = tool.get('function', {})
                tool_names.add(func.get('name', 'unknown'))
        
        print(f"Available Tools: {Fore.YELLOW}{len(tool_names)}{Style.RESET_ALL}")
        print(f"Tools: {', '.join(sorted(tool_names))}\n")
        
        # Analyze tool call frequency
        tool_call_counter = Counter()
        tool_sequences = []
        conversations_with_tools = 0
        
        for conv in self.all_data:
            conversation_tools = []
            has_tool_calls = False
            
            for msg in conv.get('messages', []):
                if msg.get('role') == 'assistant':
                    tool_calls = msg.get('tool_calls', [])
                    if tool_calls:
                        has_tool_calls = True
                        for tc in tool_calls:
                            func_name = tc.get('function', {}).get('name', 'unknown')
                            tool_call_counter[func_name] += 1
                            conversation_tools.append(func_name)
            
            if has_tool_calls:
                conversations_with_tools += 1
                tool_sequences.append(conversation_tools)
        
        print(f"Tool Call Statistics:")
        print(f"  Total tool calls: {sum(tool_call_counter.values())}")
        print(f"  Conversations with tool calls: {conversations_with_tools}/{len(self.all_data)}")
        print(f"  Average tools per conversation: {sum(tool_call_counter.values())/len(self.all_data):.2f}\n")
        
        print(f"Tool Call Frequency (Top 15):")
        for tool, count in tool_call_counter.most_common(15):
            percentage = (count / sum(tool_call_counter.values())) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {tool:40s} {count:4d} ({percentage:5.2f}%) {bar}")
        print()
        
        # Analyze tool call sequences
        print(f"Common Tool Call Sequences:")
        sequence_patterns = Counter()
        for seq in tool_sequences:
            if len(seq) >= 2:
                # Create bigrams
                for i in range(len(seq) - 1):
                    sequence_patterns[(seq[i], seq[i+1])] += 1
        
        for (tool1, tool2), count in sequence_patterns.most_common(10):
            print(f"  {tool1} → {tool2}: {count}")
        print()
        
        return {
            'total_tools': len(tool_names),
            'tool_calls': dict(tool_call_counter),
            'sequences': tool_sequences
        }
    
    def analyze_conversation_patterns(self):
        """Analyze conversation flow patterns."""
        print(f"{Fore.CYAN}{'='*80}")
        print("💬 CONVERSATION PATTERN ANALYSIS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        # Analyze authentication patterns
        auth_tools = ['find_user_id_by_email', 'find_user_id_by_name_zip']
        conversations_with_auth = 0
        auth_position = []
        
        for conv in self.all_data:
            auth_found = False
            position = 0
            for i, msg in enumerate(conv.get('messages', [])):
                if msg.get('role') == 'assistant':
                    tool_calls = msg.get('tool_calls', [])
                    for tc in tool_calls:
                        func_name = tc.get('function', {}).get('name', '')
                        if func_name in auth_tools:
                            auth_found = True
                            position = i
                            break
                if auth_found:
                    break
            
            if auth_found:
                conversations_with_auth += 1
                auth_position.append(position)
        
        print(f"Authentication Patterns:")
        print(f"  Conversations with authentication: {conversations_with_auth}/{len(self.all_data)}")
        if auth_position:
            print(f"  Average authentication message position: {sum(auth_position)/len(auth_position):.2f}")
        print()
        
        # Analyze confirmation patterns
        confirmation_keywords = ['yes', 'confirm', 'proceed', 'correct', 'go ahead']
        conversations_with_confirmation = 0
        confirmations_per_conversation = []
        
        for conv in self.all_data:
            confirmation_count = 0
            for msg in conv.get('messages', []):
                if msg.get('role') == 'user':
                    content = msg.get('content', '').lower()
                    if any(keyword in content for keyword in confirmation_keywords):
                        confirmation_count += 1
            
            if confirmation_count > 0:
                conversations_with_confirmation += 1
                confirmations_per_conversation.append(confirmation_count)
        
        print(f"Confirmation Patterns:")
        print(f"  Conversations with confirmations: {conversations_with_confirmation}/{len(self.all_data)}")
        if confirmations_per_conversation:
            print(f"  Average confirmations per conversation: {sum(confirmations_per_conversation)/len(confirmations_per_conversation):.2f}\n")
        
        # Analyze conversation topics (based on tool usage)
        topic_mapping = {
            'order_cancellation': ['cancel_pending_order'],
            'order_modification': ['modify_pending_order_items', 'modify_pending_order_address', 'modify_pending_order_payment'],
            'order_return': ['return_delivered_order_items'],
            'order_exchange': ['exchange_delivered_order_items'],
            'account_management': ['modify_user_address', 'get_user_details'],
            'order_inquiry': ['get_order_details'],
            'product_inquiry': ['get_product_details', 'list_all_product_types']
        }
        
        topic_counter = Counter()
        for conv in self.all_data:
            conv_topics = set()
            for msg in conv.get('messages', []):
                if msg.get('role') == 'assistant':
                    tool_calls = msg.get('tool_calls', [])
                    for tc in tool_calls:
                        func_name = tc.get('function', {}).get('name', '')
                        for topic, tools in topic_mapping.items():
                            if func_name in tools:
                                conv_topics.add(topic)
            
            for topic in conv_topics:
                topic_counter[topic] += 1
        
        print(f"Conversation Topics (by tool usage):")
        for topic, count in topic_counter.most_common():
            percentage = (count / len(self.all_data)) * 100
            print(f"  {topic:25s}: {count:3d} ({percentage:5.2f}%)")
        print()
        
        return {
            'auth_rate': conversations_with_auth / len(self.all_data) if self.all_data else 0,
            'topics': dict(topic_counter)
        }
    
    def analyze_content_characteristics(self):
        """Analyze content characteristics like length, complexity, etc."""
        print(f"{Fore.CYAN}{'='*80}")
        print("📝 CONTENT CHARACTERISTICS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        # Analyze message lengths
        user_message_lengths = []
        assistant_message_lengths = []
        
        for conv in self.all_data:
            for msg in conv.get('messages', []):
                content = msg.get('content', '')
                if isinstance(content, str):
                    word_count = len(content.split())
                    if msg.get('role') == 'user':
                        user_message_lengths.append(word_count)
                    elif msg.get('role') == 'assistant':
                        assistant_message_lengths.append(word_count)
        
        print(f"User Message Length (words):")
        if user_message_lengths:
            print(f"  Average: {sum(user_message_lengths)/len(user_message_lengths):.2f}")
            print(f"  Min: {min(user_message_lengths)}")
            print(f"  Max: {max(user_message_lengths)}\n")
        
        print(f"Assistant Message Length (words):")
        if assistant_message_lengths:
            print(f"  Average: {sum(assistant_message_lengths)/len(assistant_message_lengths):.2f}")
            print(f"  Min: {min(assistant_message_lengths)}")
            print(f"  Max: {max(assistant_message_lengths)}\n")
        
        # Extract common phrases from user messages
        user_phrases = []
        for conv in self.all_data:
            for msg in conv.get('messages', []):
                if msg.get('role') == 'user':
                    content = msg.get('content', '').lower()
                    # Extract sentences
                    sentences = re.split(r'[.!?]', content)
                    user_phrases.extend([s.strip() for s in sentences if s.strip()])
        
        print(f"Sample User Intents (first 10 unique):")
        unique_phrases = list(set(user_phrases))[:10]
        for phrase in unique_phrases:
            if len(phrase) > 10 and len(phrase) < 100:
                print(f"  • {phrase}")
        print()
        
        return {
            'avg_user_words': sum(user_message_lengths)/len(user_message_lengths) if user_message_lengths else 0,
            'avg_assistant_words': sum(assistant_message_lengths)/len(assistant_message_lengths) if assistant_message_lengths else 0
        }
    
    def classify_conversations_with_ai(self):
        """Use Azure OpenAI to classify conversations into categories."""
        print(f"{Fore.CYAN}{'='*80}")
        print("🤖 AI-POWERED CONVERSATION CLASSIFICATION")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        if not all([self.azure_endpoint, self.azure_api_key, self.azure_deployment]):
            print(f"{Fore.YELLOW}⚠ Azure OpenAI credentials not configured. Skipping AI classification.{Style.RESET_ALL}\n")
            return {}
        
        try:
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version="2024-08-01-preview"
            )
            
            print(f"Using Azure OpenAI deployment: {self.azure_deployment}\n")
            
            # Sample a subset for classification (to avoid rate limits)
            sample_size = min(20, len(self.all_data))
            sample_conversations = np.random.choice(len(self.all_data), sample_size, replace=False)
            
            classifications = []
            print(f"Classifying {sample_size} sample conversations...\n")
            
            for idx in sample_conversations:
                conv = self.all_data[idx]
                
                # Create a summary of the conversation
                conversation_text = ""
                for msg in conv.get('messages', []):
                    if msg.get('role') in ['user', 'assistant']:
                        content = msg.get('content', '')
                        if isinstance(content, str) and content:
                            conversation_text += f"{msg['role']}: {content[:200]}\n"
                
                # Classify using Azure OpenAI
                try:
                    response = client.chat.completions.create(
                        model=self.azure_deployment,
                        messages=[
                            {"role": "system", "content": "You are a conversation classifier for e-commerce customer service. Classify the conversation into ONE of these categories: order_modification, order_cancellation, order_return, order_exchange, account_update, product_inquiry, shipping_change, payment_change, general_inquiry. Respond with ONLY the category name."},
                            {"role": "user", "content": f"Classify this conversation:\n\n{conversation_text[:1000]}"}
                        ],
                        temperature=0.3,
                        max_tokens=50
                    )
                    
                    category = response.choices[0].message.content.strip().lower()
                    classifications.append(category)
                    
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠ Classification error: {str(e)}{Style.RESET_ALL}")
                    classifications.append('unknown')
            
            # Analyze classifications
            category_counter = Counter(classifications)
            print(f"AI Classification Results ({sample_size} samples):")
            for category, count in category_counter.most_common():
                percentage = (count / len(classifications)) * 100
                print(f"  {category:25s}: {count:3d} ({percentage:5.2f}%)")
            print()
            
            return {
                'classifications': dict(category_counter),
                'sample_size': sample_size
            }
            
        except ImportError:
            print(f"{Fore.YELLOW}⚠ openai library not installed. Run: pip install openai{Style.RESET_ALL}\n")
            return {}
        except Exception as e:
            print(f"{Fore.RED}✗ Error during AI classification: {str(e)}{Style.RESET_ALL}\n")
            return {}
    
    def perform_clustering_analysis(self):
        """Perform clustering analysis on conversations."""
        print(f"{Fore.CYAN}{'='*80}")
        print("🎯 CLUSTERING ANALYSIS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Extract conversation texts
            conversation_texts = []
            for conv in self.all_data:
                text = ""
                for msg in conv.get('messages', []):
                    if msg.get('role') in ['user', 'assistant']:
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            text += content + " "
                conversation_texts.append(text.strip())
            
            # Vectorize conversations
            print("Vectorizing conversations...")
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(conversation_texts)
            
            # Determine optimal number of clusters (use elbow method on subset)
            max_clusters = min(10, len(self.all_data) // 5)
            n_clusters = min(5, max_clusters)  # Default to 5 clusters
            
            print(f"Performing K-Means clustering with {n_clusters} clusters...\n")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Analyze clusters
            cluster_sizes = Counter(cluster_labels)
            print(f"Cluster Distribution:")
            for cluster_id, size in sorted(cluster_sizes.items()):
                percentage = (size / len(self.all_data)) * 100
                print(f"  Cluster {cluster_id}: {size:3d} conversations ({percentage:5.2f}%)")
            print()
            
            # Extract top terms for each cluster
            print(f"Top Terms per Cluster:")
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()
            
            for cluster_id in range(n_clusters):
                top_terms = [terms[ind] for ind in order_centroids[cluster_id, :5]]
                print(f"  Cluster {cluster_id}: {', '.join(top_terms)}")
            print()
            
            return {
                'n_clusters': n_clusters,
                'cluster_sizes': dict(cluster_sizes),
                'cluster_labels': cluster_labels.tolist(),
                'X': X,
                'pca_possible': True
            }
            
        except ImportError:
            print(f"{Fore.YELLOW}⚠ sklearn not installed. Run: pip install scikit-learn{Style.RESET_ALL}\n")
            return {'pca_possible': False}
        except Exception as e:
            print(f"{Fore.RED}✗ Error during clustering: {str(e)}{Style.RESET_ALL}\n")
            return {'pca_possible': False}
    
    def create_visualizations(self):
        """Create all visualization charts."""
        if not self.create_charts:
            print(f"{Fore.YELLOW}Chart creation disabled.{Style.RESET_ALL}\n")
            return
        
        print(f"{Fore.CYAN}{'='*80}")
        print("📊 CREATING VISUALIZATIONS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        # Create output directory
        output_dir = "analysis_charts/data_gen"
        os.makedirs(output_dir, exist_ok=True)
        
        # Gather statistics for charts
        stats = self.analyze_basic_statistics()
        tool_stats = self.analyze_tool_usage()
        pattern_stats = self.analyze_conversation_patterns()
        content_stats = self.analyze_content_characteristics()
        
        # Create charts
        self._create_conversation_length_chart(output_dir)
        self._create_message_role_chart(stats, output_dir)
        self._create_tool_frequency_chart(tool_stats, output_dir)
        self._create_tool_sequence_heatmap(tool_stats, output_dir)
        self._create_topic_distribution_chart(pattern_stats, output_dir)
        self._create_dataset_split_chart(stats, output_dir)
        self._create_conversation_turn_distribution(output_dir)
        self._create_tool_combinations_chart(tool_stats, output_dir)
        
        # AI classification chart
        ai_stats = self.classify_conversations_with_ai()
        if ai_stats:
            self._create_ai_classification_chart(ai_stats, output_dir)
        
        # Clustering charts
        cluster_stats = self.perform_clustering_analysis()
        if cluster_stats.get('pca_possible'):
            self._create_cluster_visualization(cluster_stats, output_dir)
            self._create_cluster_size_chart(cluster_stats, output_dir)
        
        # Summary
        print(f"\n{Fore.GREEN}✓ Created {len(self.charts_created)} charts in '{output_dir}/' directory{Style.RESET_ALL}")
        for chart in self.charts_created:
            print(f"  • {chart}")
        print()
    
    def _create_conversation_length_chart(self, output_dir):
        """Create conversation length distribution chart."""
        message_counts = []
        for conv in self.all_data:
            non_system = [m for m in conv.get('messages', []) if m.get('role') != 'system']
            message_counts.append(len(non_system))
        
        plt.figure(figsize=(12, 6))
        plt.hist(message_counts, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Messages', fontsize=12)
        plt.ylabel('Number of Conversations', fontsize=12)
        plt.title('Distribution of Conversation Lengths', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        output_path = os.path.join(output_dir, '01_conversation_length_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('01_conversation_length_distribution.png')
    
    def _create_message_role_chart(self, stats, output_dir):
        """Create message role distribution pie chart."""
        role_dist = stats.get('role_distribution', {})
        
        # Filter out system messages for clearer visualization
        filtered_roles = {k: v for k, v in role_dist.items() if k != 'system'}
        
        plt.figure(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        plt.pie(filtered_roles.values(), labels=filtered_roles.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 12})
        plt.title('Message Distribution by Role (excluding system)', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_dir, '02_message_role_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('02_message_role_distribution.png')
    
    def _create_tool_frequency_chart(self, tool_stats, output_dir):
        """Create tool call frequency bar chart."""
        tool_calls = tool_stats.get('tool_calls', {})
        
        # Get top 15 tools
        top_tools = dict(sorted(tool_calls.items(), key=lambda x: x[1], reverse=True)[:15])
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(list(top_tools.keys()), list(top_tools.values()), color='steelblue')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel('Number of Calls', fontsize=12)
        plt.ylabel('Tool Name', fontsize=12)
        plt.title('Top 15 Most Frequently Called Tools', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '03_tool_frequency.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('03_tool_frequency.png')
    
    def _create_tool_sequence_heatmap(self, tool_stats, output_dir):
        """Create tool call sequence heatmap."""
        sequences = tool_stats.get('sequences', [])
        
        # Build transition matrix
        all_tools = set()
        for seq in sequences:
            all_tools.update(seq)
        
        tool_list = sorted(list(all_tools))[:20]  # Limit to top 20 for readability
        
        if len(tool_list) < 2:
            return
        
        transition_matrix = np.zeros((len(tool_list), len(tool_list)))
        tool_to_idx = {tool: idx for idx, tool in enumerate(tool_list)}
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                if seq[i] in tool_to_idx and seq[i+1] in tool_to_idx:
                    transition_matrix[tool_to_idx[seq[i]], tool_to_idx[seq[i+1]]] += 1
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(transition_matrix, xticklabels=tool_list, yticklabels=tool_list,
                   cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Transition Count'})
        plt.xlabel('Next Tool', fontsize=12)
        plt.ylabel('Current Tool', fontsize=12)
        plt.title('Tool Call Sequence Transitions (Heatmap)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '04_tool_sequence_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('04_tool_sequence_heatmap.png')
    
    def _create_topic_distribution_chart(self, pattern_stats, output_dir):
        """Create conversation topic distribution chart."""
        topics = pattern_stats.get('topics', {})
        
        if not topics:
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(topics.keys(), topics.values(), color='coral', edgecolor='black', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Topic Category', fontsize=12)
        plt.ylabel('Number of Conversations', fontsize=12)
        plt.title('Conversation Topic Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '05_topic_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('05_topic_distribution.png')
    
    def _create_dataset_split_chart(self, stats, output_dir):
        """Create dataset split visualization."""
        splits = {
            'Training': stats.get('train_count', 0),
            'Validation': stats.get('valid_count', 0)
        }
        
        plt.figure(figsize=(8, 8))
        colors = ['#3498db', '#e74c3c']
        plt.pie(splits.values(), labels=splits.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Training vs Validation Dataset Split', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_dir, '06_dataset_split.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('06_dataset_split.png')
    
    def _create_conversation_turn_distribution(self, output_dir):
        """Create conversation turn (user-assistant exchanges) distribution."""
        turn_counts = []
        
        for conv in self.all_data:
            turns = 0
            for msg in conv.get('messages', []):
                if msg.get('role') in ['user', 'assistant']:
                    turns += 1
            turn_counts.append(turns // 2)  # Divide by 2 to get exchanges
        
        plt.figure(figsize=(12, 6))
        plt.hist(turn_counts, bins=range(1, max(turn_counts)+2), color='mediumseagreen',
                edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Turns (User-Assistant Exchanges)', fontsize=12)
        plt.ylabel('Number of Conversations', fontsize=12)
        plt.title('Distribution of Conversation Turns', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '07_conversation_turns.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('07_conversation_turns.png')
    
    def _create_tool_combinations_chart(self, tool_stats, output_dir):
        """Create chart showing most common tool combinations."""
        sequences = tool_stats.get('sequences', [])
        
        # Find common tool combinations (sets of tools used together)
        tool_sets = []
        for seq in sequences:
            unique_tools = tuple(sorted(set(seq)))
            if len(unique_tools) >= 2:
                tool_sets.append(unique_tools)
        
        tool_set_counter = Counter(tool_sets)
        top_combinations = tool_set_counter.most_common(10)
        
        if not top_combinations:
            return
        
        # Format labels
        labels = []
        values = []
        for tool_set, count in top_combinations:
            label = ' + '.join([t[:20] for t in tool_set[:3]])  # Limit to 3 tools
            if len(tool_set) > 3:
                label += '...'
            labels.append(label)
            values.append(count)
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(labels, values, color='mediumpurple')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Tool Combination', fontsize=12)
        plt.title('Top 10 Most Common Tool Combinations', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '08_tool_combinations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('08_tool_combinations.png')
    
    def _create_ai_classification_chart(self, ai_stats, output_dir):
        """Create AI classification results chart."""
        classifications = ai_stats.get('classifications', {})
        
        if not classifications:
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classifications.keys(), classifications.values(),
                      color='lightcoral', edgecolor='black', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Number of Conversations', fontsize=12)
        plt.title(f'AI-Powered Conversation Classification (n={ai_stats.get("sample_size", 0)})',
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '09_ai_classification.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('09_ai_classification.png')
    
    def _create_cluster_visualization(self, cluster_stats, output_dir):
        """Create 2D visualization of clusters using PCA."""
        try:
            from sklearn.decomposition import PCA
            
            X = cluster_stats.get('X')
            cluster_labels = np.array(cluster_stats.get('cluster_labels', []))
            
            # Reduce to 2D using PCA
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X.toarray())
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                                cmap='viridis', s=100, alpha=0.6, edgecolors='black')
            plt.colorbar(scatter, label='Cluster ID')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
            plt.title('Conversation Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, '10_cluster_visualization.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.charts_created.append('10_cluster_visualization.png')
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Could not create cluster visualization: {str(e)}{Style.RESET_ALL}")
    
    def _create_cluster_size_chart(self, cluster_stats, output_dir):
        """Create cluster size distribution chart."""
        cluster_sizes = cluster_stats.get('cluster_sizes', {})
        
        if not cluster_sizes:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([f'Cluster {k}' for k in cluster_sizes.keys()],
                      cluster_sizes.values(), color='teal', edgecolor='black', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Conversations', fontsize=12)
        plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, '11_cluster_sizes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append('11_cluster_sizes.png')
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print("🔍 SYNTHETIC DATA ANALYSIS TOOL")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        # Run all analyses
        self.analyze_basic_statistics()
        self.analyze_tool_usage()
        self.analyze_conversation_patterns()
        self.analyze_content_characteristics()
        
        # Create visualizations (includes AI classification and clustering)
        self.create_visualizations()
        
        print(f"\n{Fore.GREEN}{'='*80}")
        print("✓ ANALYSIS COMPLETE")
        print(f"{'='*80}{Style.RESET_ALL}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze synthetic fine-tuning conversation data'
    )
    parser.add_argument(
        '--train',
        default='data/datagen-ToolUse-FineTuneSupervised-train.jsonl',
        help='Path to training JSONL file'
    )
    parser.add_argument(
        '--valid',
        default='data/datagen-ToolUse-FineTuneSupervised-valid.jsonl',
        help='Path to validation JSONL file'
    )
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Disable chart creation'
    )
    
    args = parser.parse_args()
    
    analyzer = SyntheticDataAnalyzer(
        train_path=args.train,
        valid_path=args.valid,
        create_charts=not args.no_charts
    )
    
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
