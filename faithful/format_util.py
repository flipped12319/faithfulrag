import re
import string

class FormatConverter:
    def remove_brackets_and_content(input_str: str) -> str:
        cleaned_str = re.sub(r'\<*?\>', '', input_str)
        cleaned_str = re.sub(r'\[*?\]', '', cleaned_str)
        cleaned_str = ' '.join(cleaned_str.split())
        return cleaned_str

    def extract_last_answer(self,text) -> str:
        """
        从文本中提取最后一个 'answer is' 后面的内容
        支持跨行、逗号、句号、破折号等标点
        返回答案字符串，如果没找到则返回空字符串
        """
        # 匹配 answer is 后面的内容，非贪婪，直到行尾或句号结束
        pattern = r'answer\s+is\s+(.+?)(?:\.|$)'

        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)

        if matches:
            # 返回最后一个匹配
            return matches[-1].strip()
        else:
            return ""

    def clean_string(self,s: str) -> str:
        """
        去掉字符串两边的空白字符和句点 .
        """
        return s.strip().strip('.')

    def normalize_answer(self,s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        s = str(s)
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

# format=FormatConverter()
# str=format.clean_string(" 123 ..")
# print(str)


