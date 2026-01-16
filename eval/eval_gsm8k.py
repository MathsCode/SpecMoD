"""
自定义模型在 GSM8K 数据集上的评测脚本

这个脚本基于 lm-evaluation-harness 的逻辑，为你的定制模型 pipeline 评测 GSM8K 数据集。
支持两种评测模式：
1. CoT (Chain-of-Thought): 带推理过程的生成
2. Direct: 直接生成答案
"""

import re
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from model.qwen3_model_adaptor_pipeline_back import Spec_Qwen3ForCausalLM
from model.utils import storage, ShadowAdapter2, ShadowAdapter3

from transformers import Qwen3ForCausalLM
from transformers.generation import StoppingCriteria, StoppingCriteriaList
import torch

class GSM8KEvaluator:
    """GSM8K 数据集评测器"""
    
    def __init__(
        self,
        model_pipeline,
        mode: str = "cot",  # "cot" 或 "direct"
        num_fewshot: int = 8,
        limit: int = None,
        begin: int = None,
        end: int = None,
        seed: int = 42
    ):
        """
        初始化评测器
        
        参数:
            model_pipeline: 你的模型 pipeline，需要有一个 generate() 方法
            mode: 评测模式，"cot" (带推理) 或 "direct" (直接答案)
            num_fewshot: few-shot 样本数量
            limit: 限制评测样本数量，用于快速测试
            seed: 随机种子
        """
        self.model = model_pipeline
        self.mode = mode
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.seed = seed
        self.begin = begin
        self.end = end
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载数据集
        print("正在加载 GSM8K 数据集...")
        self.dataset = load_dataset("/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/eval/gsm8k", "main")
        print(f"训练集样本数: {len(self.dataset['train'])}")
        print(f"测试集样本数: {len(self.dataset['test'])}")
        
        # Few-shot 示例 (来自 gsm8k-cot.yaml)
        self.fewshot_examples = self._get_fewshot_examples()
    
    def _get_fewshot_examples(self) -> List[Dict[str, str]]:
        """获取 few-shot 示例"""
        if self.mode == "cot":
            # Chain-of-Thought 示例
            return [
                {
                    "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                    "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
                },
                {
                    "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                    "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
                },
                {
                    "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                    "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
                },
                {
                    "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                    "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
                },
                {
                    "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                    "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
                },
                {
                    "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                    "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
                },
                {
                    "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                    "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
                },
                {
                    "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                    "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
                }
            ]
        else:
            # Direct 模式: 只有答案，没有推理过程
            # 使用训练集的前几个样本作为 few-shot
            examples = []
            for i in range(min(self.num_fewshot, len(self.dataset['train']))):
                item = self.dataset['train'][i]
                examples.append({
                    "question": item['question'],
                    "answer": self._extract_answer(item['answer'])
                })
            return examples
    
    def _extract_answer(self, answer_str: str) -> str:
        """
        从答案字符串中提取数字答案
        GSM8K 的答案格式: "推理过程\n#### 答案"
        """
        # 使用 #### 分割
        parts = answer_str.split("####")
        if len(parts) == 2:
            return parts[1].strip()
        return answer_str.strip()
    
    def _build_prompt(self, question: str, use_fewshot: bool = True) -> str:
        """构建提示词"""
        if self.mode == "cot":
            # CoT 模式
            prompt_parts = []
            
            if use_fewshot and self.num_fewshot > 0:
                # 添加 few-shot 示例
                for example in self.fewshot_examples[:self.num_fewshot]:
                    prompt_parts.append(f"Q: {example['question']}\nA: {example['answer']}")
            
            # 添加当前问题
            prompt_parts.append(f"Q: {question}\nA:")
            
            return "\n\n".join(prompt_parts)
        else:
            # Direct 模式
            prompt_parts = []
            
            if use_fewshot and self.num_fewshot > 0:
                for example in self.fewshot_examples[:self.num_fewshot]:
                    prompt_parts.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
            
            prompt_parts.append(f"Question: {question}\nAnswer:")
            
            return "\n\n".join(prompt_parts)
    
    def _extract_prediction(self, generated_text: str) -> str:
        """
        从生成的文本中提取答案
        使用多种正则表达式模式尝试提取
        """
        # 模式 1: "The answer is X."
        pattern1 = r"The answer is ([\-]?[0-9\.\,]+)\."
        match = re.search(pattern1, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
        
        # 模式 2: "#### X" (GSM8K 格式)
        pattern2 = r"####\s*([\-]?[0-9\.\,]+)"
        match = re.search(pattern2, generated_text)
        if match:
            return match.group(1).replace(",", "")
        
        # 模式 3: 提取最后一个数字
        pattern3 = r"([\-]?[$0-9\.\,]{2,})|([\-]?[0-9]+)"
        matches = re.findall(pattern3, generated_text)
        if matches:
            # 返回最后一个匹配的数字
            last_match = matches[-1]
            number = last_match[0] if last_match[0] else last_match[1]
            return number.replace("$", "").replace(",", "")
        
        return ""
    
    def _normalize_number(self, num_str: str) -> str:
        """
        标准化数字字符串
        移除逗号、美元符号等
        """
        if not num_str:
            return ""
        
        # 移除常见的非数字字符
        num_str = num_str.replace(",", "").replace("$", "").strip()
        
        # 尝试转换为浮点数再转回字符串，以统一格式
        try:
            num = float(num_str)
            # 如果是整数，返回整数格式
            if num.is_integer():
                return str(int(num))
            else:
                return str(num)
        except ValueError:
            return num_str
    
    def _evaluate_single(self, question: str, ground_truth: str) -> Tuple[bool, str, str]:
        """
        评测单个样本
        
        返回:
            (是否正确, 预测答案, 生成的完整文本)
        """
        # 构建提示
        prompt = self._build_prompt(question)
        
        # 生成答案
        # 注意：这里假设你的 model_pipeline 有一个 generate 方法
        # 你需要根据实际的模型接口调整这部分代码
        if self.mode == "cot":
            # CoT 模式需要生成较长的文本
            generated = self.model.generate(
                prompt,
                max_new_tokens=1000,
                temperature=0.0000001,
                do_sample=False,
                mode = self.mode,
                stop_strings=["Q:", "</s>", "<|im_end|>"]
            )
        else:
            # Direct 模式只需生成答案
            generated = self.model.generate(
                prompt,
                max_new_tokens=300,
                temperature=0.0000001,
                do_sample=False,
                mode = self.mode,
                stop_strings=["Question:", "</s>", "<|im_end|>"]
            )
        
        # 提取预测答案
        prediction = self._extract_prediction(generated)
        
        # 标准化答案格式
        pred_normalized = self._normalize_number(prediction)
        truth_normalized = self._normalize_number(ground_truth)
        
        # 判断是否正确
        is_correct = pred_normalized == truth_normalized
        
        return is_correct, prediction, generated
    
    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """
        在指定的数据集分割上进行评测
        
        参数:
            split: 数据集分割，"train" 或 "test"
        
        返回:
            包含评测结果的字典
        """
        print(f"\n开始在 {split} 集上评测...")
        print(f"模式: {self.mode}")
        print(f"Few-shot 数量: {self.num_fewshot}")
        
        dataset = self.dataset[split]
        
        # 限制样本数量
        if self.limit:
            dataset = dataset.select(range(min(self.limit, len(dataset))))
        elif self.begin and self.end:
            dataset = dataset.select(range(max(self.begin, 0), min(self.end, len(dataset))))
        
        print(f"评测样本数: {len(dataset)}\n")
        
        correct = 0
        total = 0
        results = []
        
        # 遍历数据集
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            question = item['question']
            answer = item['answer']
            ground_truth = self._extract_answer(answer)
            
            # 评测单个样本
            is_correct, prediction, generated = self._evaluate_single(question, ground_truth)
            
            if is_correct:
                correct += 1
            total += 1
            
            # 保存结果
            results.append({
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "generated_text": generated,
                "correct": is_correct
            })
            
            # 每 100 个样本打印一次中间结果
            if (idx + 1) % 100 == 0:
                accuracy = correct / total * 100
                print(f"\n进度: {idx + 1}/{len(dataset)}, 当前准确率: {accuracy:.2f}%")
        
        # 计算最终指标
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"评测完成!")
        print(f"总样本数: {total}")
        print(f"正确数量: {correct}")
        print(f"准确率: {accuracy:.2f}%")
        print(f"{'='*50}\n")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
            "config": {
                "mode": self.mode,
                "num_fewshot": self.num_fewshot,
                "limit": self.limit,
                "seed": self.seed
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评测结果到文件"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")
    
    def show_examples(self, results: Dict[str, Any], num_examples: int = 5):
        """展示一些评测样例"""
        print(f"\n{'='*50}")
        print("评测样例展示:")
        print(f"{'='*50}\n")
        
        # 展示正确的样例
        correct_examples = [r for r in results['results'] if r['correct']]
        if correct_examples:
            print(f"【正确样例】 (展示 {min(num_examples, len(correct_examples))} 个)\n")
            for i, example in enumerate(correct_examples[:num_examples]):
                print(f"样例 {i+1}:")
                print(f"问题: {example['question']}")
                print(f"正确答案: {example['ground_truth']}")
                print(f"模型预测: {example['prediction']} ✓")
                print(f"生成文本: {example['generated_text'][:200]}...")
                print()
        
        # 展示错误的样例
        wrong_examples = [r for r in results['results'] if not r['correct']]
        if wrong_examples:
            print(f"【错误样例】 (展示 {min(num_examples, len(wrong_examples))} 个)\n")
            for i, example in enumerate(wrong_examples[:num_examples]):
                print(f"样例 {i+1}:")
                print(f"问题: {example['question']}")
                print(f"正确答案: {example['ground_truth']}")
                print(f"模型预测: {example['prediction']} ✗")
                print(f"生成文本: {example['generated_text'][:200]}...")
                print()


# ============================================================================
# 使用示例
# ============================================================================



class SpecMoDPipeline:
    """
    这是一个示例模型 pipeline 类
    你需要替换成你自己的模型实现
    """
    def __init__(self, args):
        # 在这里初始化你的模型
        # 例如: self.model = YourModel.from_pretrained(...)
        
        model_path = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Spec_Qwen3ForCausalLM.from_pretrained(model_path).half().to('cuda')
        self.LAYERS = self.model.config.num_hidden_layers
        self.adaptor = [None, ]
    
        # adaptor = nn.ModuleList([
        #         ShadowAdapter3(model.config.hidden_size, 2048) for _ in range(LAYERS)
        #     ])
        # adaptor.load_state_dict(torch.load("/inspire/hdd/project/inference-chip/xujiaming-253308120313/Paper/SpecMoD/checkpoint/adaptor/2048_finetune/final_adapters_2048_20251217_0700.pt"))
        # adaptor.half().to(model.device)
        
        
        for i in range(1, self.LAYERS):
            if i == 34:
                self.adaptor.append(None)
            else:
                layer_adaptor = ShadowAdapter3(self.model.config.hidden_size, 2048)
                layer_adaptor_weight = torch.load(f"./checkpoint/adaptor/2048/adapter_layer_{i}_2048_Model3_0.95.pt")
                layer_adaptor.load_state_dict(layer_adaptor_weight)
                layer_adaptor = layer_adaptor.half().to(self.model.device)
                self.adaptor.append(layer_adaptor)
        
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 400,
        temperature: float = 0.00000001,
        do_sample: bool = False,
        stop_strings: List[str] = None,
        mode = 'cot',
    ) -> str:
        """
        生成文本的方法
        
        参数:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样
            stop_strings: 停止字符串列表
        
        返回:
            生成的文本
        """
        # 这里是你的模型推理代码
        # 例如:
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(
        #     inputs.input_ids,
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     do_sample=do_sample,
        #     ...
        # )
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return generated_text
        
        messages = [
            {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking = True if mode == 'cot' else False, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, adaptor=self.adaptor)
        
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    # 截断到停止字符串之前
                    generated_text = generated_text[:generated_text.index(stop_str)]
                    break
        
        # 临时返回一个示例
        return generated_text


class BaselinePipeline:
    """
    这是一个示例模型 pipeline 类
    你需要替换成你自己的模型实现
    """
    def __init__(self, args):
        # 在这里初始化你的模型
        # 例如: self.model = YourModel.from_pretrained(...)
        
        model_path = f"/inspire/hdd/global_public/public_models/Qwen/{args.model}/"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen3ForCausalLM.from_pretrained(model_path).half().to('cuda')
        
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 400,
        temperature: float = 0.0,
        do_sample: bool = False,
        stop_strings: List[str] = None,
        mode = 'cot',
    ) -> str:
        """
        生成文本的方法
        
        参数:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样
            stop_strings: 停止字符串列表
        
        返回:
            生成的文本
        """
        # 这里是你的模型推理代码
        # 例如:
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(
        #     inputs.input_ids,
        #     max_new_tokens=max_new_tokens,
        #     temperature=temperature,
        #     do_sample=do_sample,
        #     ...
        # )
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return generated_text
        
        messages = [
            {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking = True if mode == 'cot' else False, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    # 截断到停止字符串之前
                    generated_text = generated_text[:generated_text.index(stop_str)]
                    break
        
        # 临时返回一个示例
        return generated_text



def main():
    """主函数 - 使用示例"""
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="Qwen3-8B")
    parser.add_argument("--optimize", "-o", action='store_true')
    parser.add_argument("--mode", type=str, default='cot')
    parser.add_argument("--limit", type=int, default = None)
    parser.add_argument("--begin", "-b", type=int, default = None)
    parser.add_argument("--end", "-e", type=int, default = None)
    args = parser.parse_args()
    # 1. 初始化你的模型 pipeline
    print("初始化模型...")
    if args.optimize:
        model_pipeline = SpecMoDPipeline(args)
    else:
        model_pipeline = BaselinePipeline(args)
    # 2. 创建评测器
    evaluator = GSM8KEvaluator(
        model_pipeline=model_pipeline,
        mode=args.mode,  # 使用 Chain-of-Thought 模式
        num_fewshot=5,  # 8-shot
        limit=args.limit,  # 限制评测 10 个样本（用于快速测试，设置为 None 评测全部）
        begin=args.begin,
        end=args.end,
        seed=42
    )
    
    # 3. 运行评测
    results = evaluator.evaluate(split="test")
    
    # 4. 展示样例
    evaluator.show_examples(results, num_examples=3)
    
    # 5. 保存结果
    if args.optimize:
        evaluator.save_results(results, f"./eval/optimize_gsm8k_results_{args.mode}_{args.limit}.json")
    else:
        evaluator.save_results(results, f"./eval/baseline_gsm8k_results_{args.mode}_{args.limit}.json")


if __name__ == "__main__":
    main()
