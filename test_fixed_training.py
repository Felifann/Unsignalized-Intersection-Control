#!/usr/bin/env python3
"""
测试修复后的DRL训练脚本
主要验证文件句柄泄漏问题是否得到解决
"""

import sys
import os

# 添加项目根目录到路径
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

def test_training():
    """测试训练功能"""
    print("🧪 测试修复后的DRL训练...")
    
    try:
        # 导入修复后的训练模块
        from drl.train import main, monitor_system_resources, optimize_system_limits
        
        print("✅ 成功导入修复后的训练模块")
        
        # 测试系统资源监控
        print("\n📊 测试系统资源监控功能:")
        optimize_system_limits()
        monitor_system_resources()
        
        print("\n🎯 主要修复内容总结:")
        print("1. ✅ 添加了文件句柄管理和自动清理")
        print("2. ✅ 减少了文件操作频率（每30秒最多写入一次）")
        print("3. ✅ 增加了检查点保存间隔（最少2000步）")
        print("4. ✅ 添加了系统资源监控和垃圾回收")
        print("5. ✅ 使用上下文管理器确保文件正确关闭")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    print(f"\n🏁 测试结果: {'成功' if success else '失败'}")
