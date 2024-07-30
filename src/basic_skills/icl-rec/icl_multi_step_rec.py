import json

import requests


def personalized_generation(
        model: str = "qwen:72b-chat-v1.5-q5_K_M",
        temperature: float = 0.1,
        top_p: float = 0.95,
        ctx: int = 13000,
        message: list = None,
        is_stream: bool = False
):
    """
    利用Ollama框架将大模型封装成类似ChatGPT接口规范的API服务，基于提供的信息为你生产个性化的回答
    :param model: ollama安装的大模型名
    :param temperature: 温度
    :param top_p: top_p
    :param ctx: 生成token长度
    :param message: 提供给大模型的对话信息
    :param is_stream: 是否流式输出大模型的应答
    :return: 基于prompt，利用大模型生成的结果
    """
    if message is None:
        message = [{}]
    url = "http://localhost:11434/api/chat"  # Ollama的api地址
    data = {
        "model": model,  # Ollama安装的模型名
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": ctx,
            "num_gpu": 128,
        },
        "messages": message,
        "stream": is_stream
    }
    if is_stream:
        response = requests.post(url=url, json=data, stream=True)
        res = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            j = json.loads(chunk.decode('utf-8'))
            res = res + j['message']['content']
            print(j['message']['content'], end="")
        return res
    else:
        response = requests.post(url=url, json=data, stream=False)
        res = json.loads(response.content)["message"]["content"]
        return res


if __name__ == "__main__":

    instruction = ("You are a recommendation system expert in the news field, providing personalized "
                   "recommendations to users based on the background information provided.")

    portrait_prompt = """
        The news I have browsed: 
        [''Wheel Of Fortune' Guest Delivers Hilarious, Off The Rails Introduction','Hard Rock Hotel New Orleans collapse: Former site engineer weighs in','Felicity Huffman begins prison sentence for college admissions scam','Outer Banks storms unearth old shipwreck from 'Graveyard of the Atlantic'','Tiffany's is selling a holiday advent calendar for $112,000','This restored 1968 Winnebago is beyond adorable','Lori Loughlin Is 'Absolutely Terrified' After Being Hit With New Charge','Bruce Willis brought Demi Moore to tears after reading her book','Celebrity kids then and now: See how they've grown','Felicity Huffman Smiles as She Begins Community Service Following Prison Release','Queen Elizabeth Finally Had Her Dream Photoshoot, Thanks to Royal Dresser Angela Kelly','Hundreds of thousands of people in California are downriver of a dam that 'could fail'','Alexandria Ocasio-Cortez 'sincerely' apologizes for blocking ex-Brooklyn politician on Twitter, settles federal lawsuit','The Rock's Gnarly Palm Is a Testament to Life Without Lifting Gloves']
        
        What features are most important to me when selecting news (Summarize my preferences briefly)?
    """

    representable_prompt = """
    You will select the news (at most 5 news) that appeal to me the most from the list of news I have browsed, based on my personal preferences. The selected news will be presented in descending order of preference. (Format: no. a browsed news).
    """

    rec_prompt = """
    Candidate Set (candidate news): 
        1. 'Browns apologize to Mason Rudolph, call Myles Garrett's actions 'unacceptable'',
        2. 'I've been writing about tiny homes for a year and finally spent 2 nights in a 300-foot home to see what it's all about   here's how it went',
        3. 'Opinion: Colin Kaepernick is about to get what he deserves: a chance',
        4. 'The Kardashians Face Backlash Over 'Insensitive' Family Food Fight in KUWTK Clip',
        5. 'THEN AND NOW: What all your favorite '90s stars are doing today',6. 'Report: Police investigating woman's death after Redskins' player Montae Nicholson took her to hospital',
        7. 'U.S. Troops Will Die If They Remain in Syria, Bashar Al-Assad Warns',
        8. '3 Indiana judges suspended after a night of drinking turned into a White Castle brawl',
        9. 'Cows swept away by Hurricane Dorian found alive   but how?',
        10. 'Surviving Santa Clarita school shooting victims on road to recovery: Latest',
        11. 'The Unlikely Star of My Family's Thanksgiving Table',
        12. 'Meghan Markle and Hillary Clinton Secretly Spent the Afternoon Together at Frogmore Cottage',
        13. 'Former North Carolina State, NBA player Anthony Grundy dies in stabbing, police say',
        14. '85 Thanksgiving Recipes You Can Make Ahead',
        15. 'Survivor Contestants Missy Byrd and Elizabeth Beisel Apologize For Their Actions',
        16. 'Pete Davidson, Kaia Gerber Are Dating, Trying to Stay 'Low Profile'',
        17. 'There's a place in the US where its been over 80 degrees since March',
        18. 'Taylor Swift Rep Hits Back at Big Machine, Claims She's Actually Owed $7.9 Million in Unpaid Royalties',
        19. 'The most talked about movie moments of the 2010s',
        20. 'Belichick mocks social media in comments on Garrett incident',
        21. '13 Reasons Why's Christian Navarro Slams Disney for Casting 'the White Guy' in The Little Mermaid',
        22. '66 Cool Tech Gifts Anyone Would Be Thrilled to Receive'
    
    
    Can you recommend 3 news from the Candidate Set similar to the selected news I've browsed (Format: [no. a browsed news : <- a candidate news >])?
    
    """

    stream = True

    step_1_message = [
        {
            "role": "system",
            "content": instruction
        },
        {
            "role": "user",
            "content": portrait_prompt
        }
    ]

    print("========== step 1 start ================")

    step_1_output = personalized_generation(message=step_1_message, is_stream=stream)

    if not stream:
        print(step_1_output)

    step_2_message = [
        {
            "role": "system",
            "content": instruction
        },
        {
            "role": "user",
            "content": portrait_prompt
        },
        {
            "role": "assistant",
            "content": step_1_output
        },
        {
            "role": "user",
            "content": representable_prompt
        }
    ]

    print("========== step 2 start ================")

    step_2_output = personalized_generation(message=step_2_message, is_stream=stream)

    if not stream:
        print(step_2_output)

    step_3_message = [
        {
            "role": "system",
            "content": instruction
        },
        {
            "role": "user",
            "content": portrait_prompt
        },
        {
            "role": "assistant",
            "content": step_1_output
        },
        {
            "role": "user",
            "content": representable_prompt
        },
        {
            "role": "assistant",
            "content": step_2_output
        },
        {
            "role": "user",
            "content": rec_prompt
        }
    ]

    print("========== step 3 start ================")

    step_3_output = personalized_generation(message=step_3_message, is_stream=stream)

    if not stream:
        print(step_3_output)
