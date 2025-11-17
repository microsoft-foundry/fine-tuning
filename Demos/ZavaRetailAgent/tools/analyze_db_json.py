"""
Database Analysis Tool for Zava Retail Database

Analyzes the db.json file to provide statistics and relationships
on products, users, and orders with visual charts.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Initialize colorama
init(autoreset=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DatabaseAnalyzer:
    """Analyze the retail database structure and relationships."""
    
    def __init__(self, db_path="data/db.json", create_charts=True):
        """Load the database file."""
        self.db_path = db_path
        self.data = None
        self.create_charts = create_charts
        self.charts_created = []
        self.load_database()
    
    def load_database(self):
        """Load the JSON database file."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        with open(self.db_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"{Fore.GREEN}✓ Loaded database from {self.db_path}{Style.RESET_ALL}\n")
    
    def analyze_products(self):
        """Analyze product catalog."""
        print(f"{Fore.CYAN}{'='*70}")
        print("📦 PRODUCT CATALOG ANALYSIS")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        products = self.data.get('products', {})
        
        total_products = len(products)
        total_variants = sum(len(p.get('variants', {})) for p in products.values())
        
        print(f"Total Product Types: {Fore.YELLOW}{total_products}{Style.RESET_ALL}")
        print(f"Total Product Variants: {Fore.YELLOW}{total_variants}{Style.RESET_ALL}")
        print(f"Average Variants per Product: {Fore.YELLOW}{total_variants/total_products:.2f}{Style.RESET_ALL}\n")
        
        # Analyze variant attributes
        all_options = []
        available_count = 0
        unavailable_count = 0
        prices = []
        
        for product in products.values():
            for variant in product.get('variants', {}).values():
                if variant.get('available'):
                    available_count += 1
                else:
                    unavailable_count += 1
                
                prices.append(variant.get('price', 0))
                options = variant.get('options', {})
                all_options.extend(options.keys())
        
        print(f"Availability:")
        print(f"  Available: {Fore.GREEN}{available_count}{Style.RESET_ALL} ({available_count*100/total_variants:.1f}%)")
        print(f"  Unavailable: {Fore.RED}{unavailable_count}{Style.RESET_ALL} ({unavailable_count*100/total_variants:.1f}%)\n")
        
        print(f"Price Range:")
        print(f"  Min: ${min(prices):.2f}")
        print(f"  Max: ${max(prices):.2f}")
        print(f"  Average: ${sum(prices)/len(prices):.2f}\n")
        
        # Common option attributes
        option_counter = Counter(all_options)
        print(f"Common Product Attributes:")
        for attr, count in option_counter.most_common(10):
            print(f"  {attr}: {count} occurrences")
        
        print()
        
        return {
            'total_products': total_products,
            'total_variants': total_variants,
            'available': available_count,
            'unavailable': unavailable_count,
            'price_stats': {'min': min(prices), 'max': max(prices), 'avg': sum(prices)/len(prices)}
        }
    
    def analyze_users(self):
        """Analyze user profiles."""
        print(f"{Fore.CYAN}{'='*70}")
        print("👥 USER ANALYSIS")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        users = self.data.get('users', {})
        
        total_users = len(users)
        print(f"Total Users: {Fore.YELLOW}{total_users}{Style.RESET_ALL}\n")
        
        # Analyze payment methods
        payment_types = Counter()
        payment_brands = Counter()
        gift_card_balances = []
        
        for user in users.values():
            payment_methods = user.get('payment_methods', {})
            for pm in payment_methods.values():
                payment_types[pm.get('source')] += 1
                if pm.get('source') == 'credit_card':
                    payment_brands[pm.get('brand')] += 1
                elif pm.get('source') == 'gift_card':
                    gift_card_balances.append(pm.get('balance', 0))
        
        print(f"Payment Methods:")
        for pm_type, count in payment_types.most_common():
            print(f"  {pm_type}: {count}")
        print()
        
        if payment_brands:
            print(f"Credit Card Brands:")
            for brand, count in payment_brands.most_common():
                print(f"  {brand}: {count}")
            print()
        
        if gift_card_balances:
            print(f"Gift Card Statistics:")
            print(f"  Total Gift Cards: {len(gift_card_balances)}")
            print(f"  Average Balance: ${sum(gift_card_balances)/len(gift_card_balances):.2f}")
            print(f"  Total Balance: ${sum(gift_card_balances):.2f}\n")
        
        # Analyze user tiers and flags
        tiers = Counter(u.get('tier', 'unknown') for u in users.values())
        abuse_flags = sum(1 for u in users.values() if u.get('abuse_flag', False))
        
        print(f"User Tiers:")
        for tier, count in tiers.most_common():
            print(f"  {tier}: {count}")
        print()
        
        print(f"Users with Abuse Flags: {Fore.RED if abuse_flags > 0 else Fore.GREEN}{abuse_flags}{Style.RESET_ALL}\n")
        
        # Analyze orders per user
        orders_per_user = [len(u.get('orders', [])) for u in users.values()]
        users_with_orders = sum(1 for count in orders_per_user if count > 0)
        
        print(f"Order Statistics:")
        print(f"  Users with Orders: {users_with_orders} ({users_with_orders*100/total_users:.1f}%)")
        print(f"  Users without Orders: {total_users - users_with_orders}")
        print(f"  Average Orders per User: {sum(orders_per_user)/total_users:.2f}")
        print(f"  Max Orders per User: {max(orders_per_user)}")
        print()
        
        # State distribution
        states = Counter(u.get('address', {}).get('state') for u in users.values())
        print(f"Top States:")
        for state, count in states.most_common(10):
            print(f"  {state}: {count}")
        print()
        
        return {
            'total_users': total_users,
            'payment_types': dict(payment_types),
            'users_with_orders': users_with_orders,
            'avg_orders_per_user': sum(orders_per_user)/total_users
        }
    
    def analyze_orders(self):
        """Analyze orders."""
        print(f"{Fore.CYAN}{'='*70}")
        print("📋 ORDER ANALYSIS")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        orders = self.data.get('orders', {})
        
        total_orders = len(orders)
        print(f"Total Orders: {Fore.YELLOW}{total_orders}{Style.RESET_ALL}\n")
        
        # Order status distribution
        statuses = Counter(o.get('status') for o in orders.values())
        print(f"Order Status Distribution:")
        for status, count in statuses.most_common():
            print(f"  {status}: {count} ({count*100/total_orders:.1f}%)")
        print()
        
        # Items per order
        items_per_order = [len(o.get('items', [])) for o in orders.values()]
        print(f"Items per Order:")
        print(f"  Min: {min(items_per_order)}")
        print(f"  Max: {max(items_per_order)}")
        print(f"  Average: {sum(items_per_order)/len(items_per_order):.2f}\n")
        
        # Order values
        order_values = []
        for order in orders.values():
            total = sum(item.get('price', 0) for item in order.get('items', []))
            order_values.append(total)
        
        print(f"Order Value Statistics:")
        print(f"  Min: ${min(order_values):.2f}")
        print(f"  Max: ${max(order_values):.2f}")
        print(f"  Average: ${sum(order_values)/len(order_values):.2f}")
        print(f"  Total Revenue: ${sum(order_values):,.2f}\n")
        
        # Payment method usage in orders
        payment_types_used = Counter()
        for order in orders.values():
            for payment in order.get('payment_history', []):
                pm_id = payment.get('payment_method_id', '')
                if 'credit_card' in pm_id:
                    payment_types_used['credit_card'] += 1
                elif 'paypal' in pm_id:
                    payment_types_used['paypal'] += 1
                elif 'gift_card' in pm_id:
                    payment_types_used['gift_card'] += 1
        
        print(f"Payment Methods Used in Orders:")
        for pm_type, count in payment_types_used.most_common():
            print(f"  {pm_type}: {count}")
        print()
        
        # Fulfillment analysis
        orders_with_fulfillments = sum(1 for o in orders.values() if o.get('fulfillments'))
        print(f"Fulfillment Status:")
        print(f"  Orders with Fulfillments: {orders_with_fulfillments} ({orders_with_fulfillments*100/total_orders:.1f}%)")
        print(f"  Orders without Fulfillments: {total_orders - orders_with_fulfillments}\n")
        
        return {
            'total_orders': total_orders,
            'statuses': dict(statuses),
            'avg_order_value': sum(order_values)/len(order_values),
            'total_revenue': sum(order_values)
        }
    
    def analyze_relationships(self):
        """Analyze relationships between entities."""
        print(f"{Fore.CYAN}{'='*70}")
        print("🔗 ENTITY RELATIONSHIPS")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        users = self.data.get('users', {})
        orders = self.data.get('orders', {})
        products = self.data.get('products', {})
        
        # User -> Order relationship
        user_order_mapping = defaultdict(list)
        for user_id, user in users.items():
            user_order_mapping[user_id] = user.get('orders', [])
        
        print(f"User → Order Relationships:")
        print(f"  Total User-Order Links: {sum(len(orders) for orders in user_order_mapping.values())}")
        
        # Verify order references
        all_order_ids_from_users = set()
        for orders_list in user_order_mapping.values():
            all_order_ids_from_users.update(orders_list)
        
        actual_order_ids = set(orders.keys())
        orphaned_orders = actual_order_ids - all_order_ids_from_users
        missing_orders = all_order_ids_from_users - actual_order_ids
        
        if orphaned_orders:
            print(f"  {Fore.YELLOW}⚠ Orphaned Orders (no user reference): {len(orphaned_orders)}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}✓ All orders have user references{Style.RESET_ALL}")
        
        if missing_orders:
            print(f"  {Fore.RED}⚠ Missing Orders (referenced but not found): {len(missing_orders)}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}✓ All user order references are valid{Style.RESET_ALL}")
        print()
        
        # Order -> Product relationship
        product_usage = Counter()
        for order in orders.values():
            for item in order.get('items', []):
                product_id = item.get('product_id')
                if product_id:
                    product_usage[product_id] += 1
        
        print(f"Order → Product Relationships:")
        print(f"  Unique Products Ordered: {len(product_usage)}/{len(products)}")
        print(f"  Products Never Ordered: {len(products) - len(product_usage)}\n")
        
        print(f"Top 10 Most Ordered Products:")
        product_names = {p['product_id']: p['name'] for p in products.values()}
        for product_id, count in product_usage.most_common(10):
            product_name = product_names.get(product_id, 'Unknown')
            print(f"  {product_name}: {count} orders")
        print()
        
        return {
            'user_order_links': sum(len(orders) for orders in user_order_mapping.values()),
            'orphaned_orders': len(orphaned_orders),
            'products_ordered': len(product_usage),
            'products_never_ordered': len(products) - len(product_usage),
            'product_usage': product_usage
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the database."""
        if not self.create_charts:
            return
        
        print(f"{Fore.CYAN}{'='*70}")
        print("📊 GENERATING VISUALIZATIONS")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        # Create output directory
        output_dir = "analysis_charts/db_json"
        os.makedirs(output_dir, exist_ok=True)
        
        products = self.data.get('products', {})
        users = self.data.get('users', {})
        orders = self.data.get('orders', {})
        
        # 1. Product Price Distribution
        self._create_price_distribution_chart(products, output_dir)
        
        # 2. Order Status Distribution
        self._create_order_status_chart(orders, output_dir)
        
        # 3. User State Distribution
        self._create_user_location_chart(users, output_dir)
        
        # 4. Payment Method Distribution
        self._create_payment_method_chart(users, orders, output_dir)
        
        # 5. Orders per User Distribution
        self._create_orders_per_user_chart(users, output_dir)
        
        # 6. Order Value Distribution
        self._create_order_value_chart(orders, output_dir)
        
        # 7. Top Products
        self._create_top_products_chart(products, orders, output_dir)
        
        # 8. Product Availability
        self._create_product_availability_chart(products, output_dir)
        
        # === NEW CREATIVE CHARTS ===
        
        # 9. User Spending Patterns
        self._create_user_spending_patterns_chart(users, orders, output_dir)
        
        # 10. Order Size Distribution (items per order)
        self._create_order_size_heatmap(orders, output_dir)
        
        # 11. Product Category Popularity
        self._create_product_category_analysis(products, orders, output_dir)
        
        # 12. Customer Segmentation (RFM-style)
        self._create_customer_segmentation_chart(users, orders, output_dir)
        
        # 13. Payment Method by Order Value
        self._create_payment_value_correlation(orders, output_dir)
        
        # 14. Time-based Order Analysis (if timestamp available)
        self._create_order_timeline_chart(orders, output_dir)
        
        # 15. Product Variant Diversity
        self._create_product_variant_diversity(products, output_dir)
        
        # 16. User Tier Analysis
        self._create_user_tier_analysis(users, orders, output_dir)
        
        print(f"\n{Fore.GREEN}✓ All charts saved to '{output_dir}/' directory{Style.RESET_ALL}")
        print(f"  Total charts created: {len(self.charts_created)}\n")
        
        return self.charts_created
    
    def _create_price_distribution_chart(self, products, output_dir):
        """Create price distribution histogram."""
        prices = []
        for product in products.values():
            for variant in product.get('variants', {}).values():
                prices.append(variant.get('price', 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(prices, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Price ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Product Price Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot
        ax2.boxplot(prices, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title('Price Distribution (Box Plot)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '01_price_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_order_status_chart(self, orders, output_dir):
        """Create order status pie chart."""
        statuses = Counter(o.get('status') for o in orders.values())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette('Set2', len(statuses))
        
        wedges, texts, autotexts = ax.pie(
            statuses.values(),
            labels=statuses.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Order Status Distribution', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '02_order_status.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_user_location_chart(self, users, output_dir):
        """Create user state distribution chart."""
        states = Counter(u.get('address', {}).get('state') for u in users.values())
        top_states = dict(states.most_common(15))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(top_states)), list(top_states.values()),
                      color=sns.color_palette('viridis', len(top_states)))
        ax.set_xticks(range(len(top_states)))
        ax.set_xticklabels(list(top_states.keys()), rotation=45, ha='right')
        ax.set_xlabel('State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 States by User Count', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '03_user_locations.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_payment_method_chart(self, users, orders, output_dir):
        """Create payment method comparison chart."""
        # User payment methods
        user_pm = Counter()
        for user in users.values():
            for pm in user.get('payment_methods', {}).values():
                user_pm[pm.get('source')] += 1
        
        # Order payment usage
        order_pm = Counter()
        for order in orders.values():
            for payment in order.get('payment_history', []):
                pm_id = payment.get('payment_method_id', '')
                if 'credit_card' in pm_id:
                    order_pm['credit_card'] += 1
                elif 'paypal' in pm_id:
                    order_pm['paypal'] += 1
                elif 'gift_card' in pm_id:
                    order_pm['gift_card'] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # User payment methods
        colors1 = sns.color_palette('Set3', len(user_pm))
        ax1.bar(user_pm.keys(), user_pm.values(), color=colors1, edgecolor='black')
        ax1.set_xlabel('Payment Method', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('Payment Methods Registered by Users', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # Order payment usage
        colors2 = sns.color_palette('Set2', len(order_pm))
        ax2.bar(order_pm.keys(), order_pm.values(), color=colors2, edgecolor='black')
        ax2.set_xlabel('Payment Method', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_title('Payment Methods Used in Orders', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '04_payment_methods.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_orders_per_user_chart(self, users, output_dir):
        """Create orders per user distribution."""
        orders_per_user = [len(u.get('orders', [])) for u in users.values()]
        order_counts = Counter(orders_per_user)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = sorted(order_counts.keys())
        y = [order_counts[i] for i in x]
        
        bars = ax.bar(x, y, color='coral', edgecolor='darkred', alpha=0.7)
        ax.set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Orders per User', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '05_orders_per_user.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_order_value_chart(self, orders, output_dir):
        """Create order value distribution."""
        order_values = []
        for order in orders.values():
            total = sum(item.get('price', 0) for item in order.get('items', []))
            order_values.append(total)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(order_values, bins=40, color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
        ax1.set_xlabel('Order Value ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Order Value Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Violin plot
        parts = ax2.violinplot([order_values], vert=True, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightgreen')
            pc.set_alpha(0.7)
        ax2.set_ylabel('Order Value ($)', fontsize=12)
        ax2.set_title('Order Value (Violin Plot)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks([])
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '06_order_values.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_top_products_chart(self, products, orders, output_dir):
        """Create top products chart."""
        product_usage = Counter()
        for order in orders.values():
            for item in order.get('items', []):
                product_id = item.get('product_id')
                if product_id:
                    product_usage[product_id] += 1
        
        product_names = {p['product_id']: p['name'] for p in products.values()}
        top_10 = product_usage.most_common(10)
        
        names = [product_names.get(pid, 'Unknown')[:20] for pid, _ in top_10]
        counts = [count for _, count in top_10]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette('rocket', len(names))
        bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='black')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Most Ordered Products', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '07_top_products.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_product_availability_chart(self, products, output_dir):
        """Create product availability chart."""
        available = 0
        unavailable = 0
        
        for product in products.values():
            for variant in product.get('variants', {}).values():
                if variant.get('available'):
                    available += 1
                else:
                    unavailable += 1
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#2ecc71', '#e74c3c']
        sizes = [available, unavailable]
        labels = [f'Available\n({available})', f'Unavailable\n({unavailable})']
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 13, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax.set_title('Product Variant Availability', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '08_product_availability.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_user_spending_patterns_chart(self, users, orders, output_dir):
        """Create user spending patterns scatter plot."""
        user_data = []
        for user_id, user in users.items():
            order_ids = user.get('orders', [])
            total_spent = 0
            num_orders = len(order_ids)
            
            for order_id in order_ids:
                if order_id in orders:
                    order = orders[order_id]
                    total_spent += sum(item.get('price', 0) for item in order.get('items', []))
            
            if num_orders > 0:
                user_data.append({
                    'user_id': user_id,
                    'num_orders': num_orders,
                    'total_spent': total_spent,
                    'avg_order_value': total_spent / num_orders,
                    'tier': user.get('tier', 'standard')
                })
        
        df = pd.DataFrame(user_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter: Orders vs Total Spent
        tier_colors = {'standard': 'steelblue', 'premium': 'gold', 'vip': 'crimson'}
        for tier in df['tier'].unique():
            tier_data = df[df['tier'] == tier]
            ax1.scatter(tier_data['num_orders'], tier_data['total_spent'], 
                       alpha=0.6, s=100, label=tier.title(), 
                       color=tier_colors.get(tier, 'gray'))
        
        ax1.set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Spent ($)', fontsize=12, fontweight='bold')
        ax1.set_title('User Spending Patterns by Tier', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram: Average Order Value Distribution
        ax2.hist(df['avg_order_value'], bins=30, color='teal', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Average Order Value ($)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Average Order Value per User', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '09_user_spending_patterns.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_order_size_heatmap(self, orders, output_dir):
        """Create heatmap showing order size vs status."""
        order_data = []
        for order in orders.values():
            num_items = len(order.get('items', []))
            status = order.get('status', 'unknown')
            total_value = sum(item.get('price', 0) for item in order.get('items', []))
            order_data.append({
                'num_items': num_items,
                'status': status,
                'total_value': total_value
            })
        
        df = pd.DataFrame(order_data)
        
        # Create bins for items
        df['item_range'] = pd.cut(df['num_items'], bins=[0, 1, 2, 3, 5, 10, 100], 
                                   labels=['1', '2', '3', '4-5', '6-10', '10+'])
        
        # Pivot table for heatmap
        pivot = df.groupby(['item_range', 'status']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                    cbar_kws={'label': 'Number of Orders'}, linewidths=0.5)
        ax.set_xlabel('Order Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Items in Order', fontsize=12, fontweight='bold')
        ax.set_title('Order Size Distribution by Status (Heatmap)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, '10_order_size_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_product_category_analysis(self, products, orders, output_dir):
        """Create product category popularity and revenue analysis."""
        # Extract product usage and revenue
        product_stats = defaultdict(lambda: {'orders': 0, 'revenue': 0, 'variants': 0})
        
        for product in products.values():
            product_id = product['product_id']
            product_name = product['name']
            product_stats[product_name]['variants'] = len(product.get('variants', {}))
        
        for order in orders.values():
            for item in order.get('items', []):
                product_name = item.get('name', 'Unknown')
                price = item.get('price', 0)
                product_stats[product_name]['orders'] += 1
                product_stats[product_name]['revenue'] += price
        
        # Convert to DataFrame
        data = []
        for name, stats in product_stats.items():
            if stats['orders'] > 0:  # Only products that were ordered
                data.append({
                    'product': name,
                    'orders': stats['orders'],
                    'revenue': stats['revenue'],
                    'variants': stats['variants']
                })
        
        df = pd.DataFrame(data).sort_values('revenue', ascending=False).head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Revenue by product
        colors1 = sns.color_palette('Spectral', len(df))
        bars1 = ax1.barh(df['product'], df['revenue'], color=colors1, edgecolor='black')
        ax1.set_xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 15 Products by Revenue', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2., f' ${width:,.0f}',
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Bubble chart: Orders vs Revenue (bubble size = variants)
        ax2.scatter(df['orders'], df['revenue'], s=df['variants']*50, 
                   alpha=0.6, c=range(len(df)), cmap='viridis', edgecolors='black')
        ax2.set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Product Performance\n(Bubble size = # of variants)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '11_product_category_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_customer_segmentation_chart(self, users, orders, output_dir):
        """Create customer segmentation based on order frequency and value."""
        user_segments = []
        
        for user_id, user in users.items():
            order_ids = user.get('orders', [])
            num_orders = len(order_ids)
            total_spent = 0
            
            for order_id in order_ids:
                if order_id in orders:
                    order = orders[order_id]
                    total_spent += sum(item.get('price', 0) for item in order.get('items', []))
            
            # Segment customers
            if num_orders == 0:
                segment = 'Inactive'
            elif num_orders == 1:
                segment = 'One-time'
            elif num_orders <= 3:
                segment = 'Occasional'
            elif num_orders <= 5:
                segment = 'Regular'
            else:
                segment = 'Loyal'
            
            user_segments.append({
                'segment': segment,
                'num_orders': num_orders,
                'total_spent': total_spent,
                'tier': user.get('tier', 'standard')
            })
        
        df = pd.DataFrame(user_segments)
        segment_counts = df['segment'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Pie chart of segments
        colors = sns.color_palette('Set3', len(segment_counts))
        wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
        ax1.set_title('Customer Segmentation\n(by Order Frequency)', fontsize=14, fontweight='bold')
        
        # Box plot of spending by segment
        segment_order = ['Inactive', 'One-time', 'Occasional', 'Regular', 'Loyal']
        active_segments = [s for s in segment_order if s in df['segment'].unique()]
        
        data_to_plot = [df[df['segment'] == seg]['total_spent'].values 
                        for seg in active_segments]
        
        bp = ax2.boxplot(data_to_plot, labels=active_segments, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        
        ax2.set_xlabel('Customer Segment', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Spent ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Spending Distribution by Customer Segment', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '12_customer_segmentation.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_payment_value_correlation(self, orders, output_dir):
        """Analyze payment method preference by order value."""
        payment_data = []
        
        for order in orders.values():
            total_value = sum(item.get('price', 0) for item in order.get('items', []))
            
            for payment in order.get('payment_history', []):
                pm_id = payment.get('payment_method_id', '')
                pm_type = 'other'
                if 'credit_card' in pm_id:
                    pm_type = 'Credit Card'
                elif 'paypal' in pm_id:
                    pm_type = 'PayPal'
                elif 'gift_card' in pm_id:
                    pm_type = 'Gift Card'
                
                payment_data.append({
                    'payment_method': pm_type,
                    'order_value': total_value,
                    'status': order.get('status', 'unknown')
                })
        
        df = pd.DataFrame(payment_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Violin plot
        sns.violinplot(data=df, x='payment_method', y='order_value', 
                      palette='muted', ax=ax)
        ax.set_xlabel('Payment Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Order Value ($)', fontsize=12, fontweight='bold')
        ax.set_title('Order Value Distribution by Payment Method', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean markers
        means = df.groupby('payment_method')['order_value'].mean()
        for i, (method, mean_val) in enumerate(means.items()):
            ax.plot(i, mean_val, 'D', color='red', markersize=10, 
                   label='Mean' if i == 0 else '')
        
        ax.legend()
        plt.tight_layout()
        
        filename = os.path.join(output_dir, '13_payment_value_correlation.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_order_timeline_chart(self, orders, output_dir):
        """Create order status flow visualization."""
        status_counts = Counter(o.get('status') for o in orders.values())
        
        # Create a flow/funnel chart
        statuses = ['pending', 'processed', 'delivered', 'cancelled']
        counts = [status_counts.get(s, 0) for s in statuses]
        colors_map = {'pending': '#FFA500', 'processed': '#4169E1', 
                     'delivered': '#32CD32', 'cancelled': '#DC143C'}
        colors = [colors_map.get(s, 'gray') for s in statuses]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Funnel chart
        y_pos = np.arange(len(statuses))
        bars = ax1.barh(y_pos, counts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([s.title() for s in statuses], fontsize=11, fontweight='bold')
        ax1.set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
        ax1.set_title('Order Status Funnel', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax1.text(width/2, bar.get_y() + bar.get_height()/2., 
                    f'{count}', ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
        
        # Items distribution by status
        items_by_status = defaultdict(list)
        for order in orders.values():
            status = order.get('status', 'unknown')
            num_items = len(order.get('items', []))
            items_by_status[status].append(num_items)
        
        data_to_plot = [items_by_status.get(s, [0]) for s in statuses if s in items_by_status]
        labels_to_plot = [s.title() for s in statuses if s in items_by_status]
        
        bp = ax2.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], [colors_map.get(s, 'gray') for s in statuses if s in items_by_status]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Order Status', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Items', fontsize=12, fontweight='bold')
        ax2.set_title('Items per Order by Status', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '14_order_status_flow.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_product_variant_diversity(self, products, output_dir):
        """Analyze product variant diversity and options."""
        variant_counts = []
        option_diversity = defaultdict(set)
        
        for product in products.values():
            product_name = product['name']
            variants = product.get('variants', {})
            variant_counts.append(len(variants))
            
            for variant in variants.values():
                options = variant.get('options', {})
                for key, value in options.items():
                    option_diversity[key].add(value)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Distribution of variants per product
        ax1.hist(variant_counts, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of Variants per Product', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax1.set_title('Product Variant Diversity', fontsize=14, fontweight='bold')
        ax1.axvline(np.mean(variant_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(variant_counts):.1f}')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Option attribute diversity
        option_names = list(option_diversity.keys())[:10]  # Top 10
        option_counts = [len(option_diversity[opt]) for opt in option_names]
        
        colors2 = sns.color_palette('plasma', len(option_names))
        bars = ax2.bar(range(len(option_names)), option_counts, color=colors2, edgecolor='black')
        ax2.set_xticks(range(len(option_names)))
        ax2.set_xticklabels(option_names, rotation=45, ha='right')
        ax2.set_xlabel('Option Attribute', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Unique Values', fontsize=12, fontweight='bold')
        ax2.set_title('Product Option Diversity', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, '15_product_variant_diversity.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def _create_user_tier_analysis(self, users, orders, output_dir):
        """Analyze user behavior by tier."""
        tier_data = defaultdict(lambda: {'users': 0, 'total_orders': 0, 'total_revenue': 0, 'avg_order_value': []})
        
        for user_id, user in users.items():
            tier = user.get('tier', 'standard')
            tier_data[tier]['users'] += 1
            
            order_ids = user.get('orders', [])
            tier_data[tier]['total_orders'] += len(order_ids)
            
            for order_id in order_ids:
                if order_id in orders:
                    order = orders[order_id]
                    order_value = sum(item.get('price', 0) for item in order.get('items', []))
                    tier_data[tier]['total_revenue'] += order_value
                    tier_data[tier]['avg_order_value'].append(order_value)
        
        # Prepare data
        tiers = list(tier_data.keys())
        user_counts = [tier_data[t]['users'] for t in tiers]
        avg_orders = [tier_data[t]['total_orders'] / tier_data[t]['users'] if tier_data[t]['users'] > 0 else 0 for t in tiers]
        avg_revenue = [tier_data[t]['total_revenue'] / tier_data[t]['users'] if tier_data[t]['users'] > 0 else 0 for t in tiers]
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # User count by tier
        colors1 = sns.color_palette('pastel', len(tiers))
        ax1.bar(tiers, user_counts, color=colors1, edgecolor='black')
        ax1.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
        ax1.set_title('Users by Tier', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Average orders per user by tier
        colors2 = sns.color_palette('muted', len(tiers))
        ax2.bar(tiers, avg_orders, color=colors2, edgecolor='black')
        ax2.set_ylabel('Average Orders per User', fontsize=11, fontweight='bold')
        ax2.set_title('Average Orders by Tier', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Revenue comparison
        x = np.arange(len(tiers))
        width = 0.35
        
        ax3.bar(x - width/2, avg_revenue, width, label='Avg Revenue per User', 
               color='lightcoral', edgecolor='black')
        ax3.bar(x + width/2, [tier_data[t]['total_revenue'] for t in tiers], width,
               label='Total Revenue', color='lightseagreen', edgecolor='black')
        
        ax3.set_xlabel('User Tier', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
        ax3.set_title('Revenue Analysis by User Tier', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([t.title() for t in tiers])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle('User Tier Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        filename = os.path.join(output_dir, '16_user_tier_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.charts_created.append(filename)
        print(f"  ✓ Created: {filename}")
    
    def generate_summary(self):
        """Generate a complete database summary."""
        print(f"{Fore.GREEN}{Style.BRIGHT}{'='*70}")
        print("📊 ZAVA RETAIL DATABASE SUMMARY")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        product_stats = self.analyze_products()
        user_stats = self.analyze_users()
        order_stats = self.analyze_orders()
        relationship_stats = self.analyze_relationships()
        
        # Create visualizations
        if self.create_charts:
            charts = self.create_visualizations()
        
        print(f"{Fore.GREEN}{Style.BRIGHT}{'='*70}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        return {
            'products': product_stats,
            'users': user_stats,
            'orders': order_stats,
            'relationships': relationship_stats,
            'charts': self.charts_created if self.create_charts else []
        }


def main():
    """Run the database analysis."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Zava Retail Database')
    parser.add_argument('--db', type=str, default='data/db.json', help='Path to db.json file')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    
    args = parser.parse_args()
    
    try:
        analyzer = DatabaseAnalyzer(args.db, create_charts=not args.no_charts)
        summary = analyzer.generate_summary()
        
    except FileNotFoundError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
