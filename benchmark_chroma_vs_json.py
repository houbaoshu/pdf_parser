#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比测试：Chroma vs JSON
比较两种存储方式的查询性能
"""

import time
import logging
from typing import List, Dict, Any
import statistics

# 设置日志
logging.basicConfig(
    level=logging.WARNING,  # 减少日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """性能测试运行器"""

    def __init__(self):
        self.test_queries = [
            "GEA设备技术参数",
            "设备结构图",
            "安装要求",
            "维护保养",
            "技术规格",
            "操作说明",
            "安全注意事项",
            "故障排除",
            "设备配置",
            "性能指标"
        ]

    def benchmark_json_agent(self) -> Dict[str, Any]:
        """
        测试JSON版本的Agent性能

        Returns:
            性能统计字典
        """
        print("\n" + "=" * 80)
        print("测试JSON版本Agent")
        print("=" * 80)

        try:
            from gea_qa_agent import GEAQAAgent

            # 初始化
            print("初始化Agent...")
            init_start = time.time()
            agent = GEAQAAgent()
            if not agent.initialize():
                print("初始化失败")
                return {}
            init_time = time.time() - init_start
            print(f"初始化耗时: {init_time:.2f}秒")

            # 执行查询测试
            query_times = []
            print(f"\n执行 {len(self.test_queries)} 次查询...")

            for i, query in enumerate(self.test_queries, 1):
                start = time.time()
                result = agent.query(query, query_type="text", top_k=5)
                elapsed = time.time() - start
                query_times.append(elapsed)

                print(f"  查询 {i}/{len(self.test_queries)}: '{query}' - {elapsed:.3f}秒 - 找到{len(result.search_results)}个结果")

            # 统计
            avg_time = statistics.mean(query_times)
            median_time = statistics.median(query_times)
            min_time = min(query_times)
            max_time = max(query_times)
            std_dev = statistics.stdev(query_times) if len(query_times) > 1 else 0

            stats = {
                "name": "JSON版本",
                "init_time": init_time,
                "query_times": query_times,
                "avg_query_time": avg_time,
                "median_query_time": median_time,
                "min_query_time": min_time,
                "max_query_time": max_time,
                "std_dev": std_dev,
                "total_chunks": len(agent.loader.chunks)
            }

            return stats

        except Exception as e:
            print(f"JSON版本测试失败: {str(e)}")
            return {}

    def benchmark_chroma_agent(self) -> Dict[str, Any]:
        """
        测试Chroma版本的Agent性能

        Returns:
            性能统计字典
        """
        print("\n" + "=" * 80)
        print("测试Chroma版本Agent")
        print("=" * 80)

        try:
            from gea_qa_agent_chroma import GEAQAAgentChroma

            # 初始化
            print("初始化Agent...")
            init_start = time.time()
            agent = GEAQAAgentChroma()
            if not agent.initialize():
                print("初始化失败")
                return {}
            init_time = time.time() - init_start
            print(f"初始化耗时: {init_time:.2f}秒")

            # 执行查询测试
            query_times = []
            print(f"\n执行 {len(self.test_queries)} 次查询...")

            for i, query in enumerate(self.test_queries, 1):
                start = time.time()
                result = agent.query(query, query_type="text", top_k=5)
                elapsed = time.time() - start
                query_times.append(elapsed)

                print(f"  查询 {i}/{len(self.test_queries)}: '{query}' - {elapsed:.3f}秒 - 找到{len(result.search_results)}个结果")

            # 统计
            avg_time = statistics.mean(query_times)
            median_time = statistics.median(query_times)
            min_time = min(query_times)
            max_time = max(query_times)
            std_dev = statistics.stdev(query_times) if len(query_times) > 1 else 0

            # 获取chunks数量
            stats_info = agent.retriever.get_statistics()

            stats = {
                "name": "Chroma版本",
                "init_time": init_time,
                "query_times": query_times,
                "avg_query_time": avg_time,
                "median_query_time": median_time,
                "min_query_time": min_time,
                "max_query_time": max_time,
                "std_dev": std_dev,
                "total_chunks": stats_info.get("total_chunks", 0)
            }

            return stats

        except Exception as e:
            print(f"Chroma版本测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def print_comparison(self, json_stats: Dict[str, Any], chroma_stats: Dict[str, Any]):
        """
        打印对比结果

        Args:
            json_stats: JSON版本统计
            chroma_stats: Chroma版本统计
        """
        print("\n" + "=" * 80)
        print("性能对比结果")
        print("=" * 80)

        if not json_stats or not chroma_stats:
            print("缺少统计数据，无法对比")
            return

        # 初始化时间对比
        print("\n1. 初始化时间:")
        print(f"  JSON版本:   {json_stats['init_time']:.2f}秒")
        print(f"  Chroma版本: {chroma_stats['init_time']:.2f}秒")
        init_speedup = json_stats['init_time'] / chroma_stats['init_time'] if chroma_stats['init_time'] > 0 else 0
        print(f"  速度提升:   {init_speedup:.2f}x")

        # 查询时间对比
        print("\n2. 查询性能:")
        print(f"  {'指标':<20} {'JSON版本':>15} {'Chroma版本':>15} {'速度提升':>12}")
        print("-" * 70)

        metrics = [
            ("平均查询时间", "avg_query_time"),
            ("中位数查询时间", "median_query_time"),
            ("最快查询时间", "min_query_time"),
            ("最慢查询时间", "max_query_time"),
            ("标准差", "std_dev")
        ]

        for label, key in metrics:
            json_val = json_stats[key]
            chroma_val = chroma_stats[key]
            speedup = json_val / chroma_val if chroma_val > 0 else 0

            print(f"  {label:<20} {json_val:>12.3f}秒 {chroma_val:>12.3f}秒 {speedup:>10.2f}x")

        # 数据规模
        print("\n3. 数据规模:")
        print(f"  JSON版本 chunks:   {json_stats.get('total_chunks', 0)}")
        print(f"  Chroma版本 chunks: {chroma_stats.get('total_chunks', 0)}")

        # 总结
        print("\n" + "=" * 80)
        print("总结:")
        avg_speedup = json_stats['avg_query_time'] / chroma_stats['avg_query_time'] if chroma_stats['avg_query_time'] > 0 else 0

        if avg_speedup > 1:
            print(f"✅ Chroma版本平均查询速度快 {avg_speedup:.2f}x")
        elif avg_speedup < 1:
            print(f"⚠️  JSON版本平均查询速度快 {1/avg_speedup:.2f}x")
        else:
            print("⚖️  两个版本速度相当")

        if chroma_stats['init_time'] < json_stats['init_time']:
            print(f"✅ Chroma版本初始化速度快 {init_speedup:.2f}x")
        else:
            print(f"⚠️  JSON版本初始化速度快 {1/init_speedup:.2f}x")

        print("\n推荐:")
        if avg_speedup > 1.2:
            print("✅ 推荐使用Chroma版本（性能更好）")
        elif avg_speedup < 0.8:
            print("⚠️  JSON版本性能更好（少见情况）")
        else:
            print("⚖️  两个版本性能相当，Chroma版本扩展性更好")

        print("=" * 80)

    def run_full_benchmark(self):
        """运行完整的性能测试"""
        print("=" * 80)
        print("RAG性能测试：JSON vs Chroma")
        print("=" * 80)
        print(f"测试查询数量: {len(self.test_queries)}")
        print(f"每次查询返回: top_k=5")

        # 测试JSON版本
        json_stats = self.benchmark_json_agent()

        # 测试Chroma版本
        chroma_stats = self.benchmark_chroma_agent()

        # 打印对比结果
        self.print_comparison(json_stats, chroma_stats)


def main():
    """主函数"""
    runner = BenchmarkRunner()
    runner.run_full_benchmark()


if __name__ == "__main__":
    main()
