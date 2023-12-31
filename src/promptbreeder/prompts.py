from typing import Optional

THINKING_STYLES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this problem and its solutions?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: Analyzing the problem from different perspectives, questioning assumptions, and evaluating evidence.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem.",
    "Seek input and collaboration from others, emphasizing teamwork and open communication.",
    "Use systems thinking to consider the problem as part of a larger system.",
    "Use Risk Analysis to evaluate potential risks and trade-offs of different solutions.",
    "Use Reflective Thinking for introspection and self-reflection.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? What were the outcomes?",
    "What are the potential obstacles or challenges in solving this problem?",
    "Are there any relevant data or information for insights into the problem?",
    "Are there any stakeholders or individuals directly affected by the problem?",
    "What resources are needed to tackle the problem effectively?",
    "How can progress or success in solving the problem be measured?",
    "What indicators or metrics can be used?",
    "Is the problem technical, practical, conceptual, or theoretical?",
    "Does the problem involve a physical constraint?",
    "Is the problem related to human behavior?",
    "Does the problem involve decision-making or planning under uncertainty?",
    "Is the problem an analytical one requiring data analysis or modeling?",
    "Is the problem a design challenge requiring creative solutions?",
    "Does the problem require addressing systemic or structural issues?",
    "Is the problem time-sensitive or urgent?",
    "What kinds of solution typically are produced for this kind of problem?",
    "Given the problem and current best solution, guess about other possible solutions.",
    "Imagine the current best solution is wrong, what other ways to think about the problem?",
    "The best way to modify the current best solution based on problem specification.",
    "Ignoring the current best solution, create a new solution.",
    "Think step by step.",
    "Make a step by step plan and implement it with good notion and explanation.",
]

MUTATION_PROMPTS = [
    "Modify the following instruction creatively, giving some advice on how to solve it.",
    "Just change this instruction to make it more fun, think WELL outside the box.",
    "Modify this instruction in a way that no self-respecting LLM would!",
    'How would you encourage someone and help find a "hack" or "shortcut" on the instruction?',
    "How would you help an LLM to follow the instruction?",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants, as if you were explaining it to a child.",
    "Keep the original instruction, but add some hints to the end.",
    "Add an example to the instruction that shows what it looks like to do it well.",
    "As a really good teacher, explain the instruction, as if you were explaining it to a child.",
    "Imagine you need to follow this instruction. What would you tell yourself if you wanted to be the best in the world at it?",
    "How would someone with derailment follow this instruction?",
    "Don't think about the instruction at all, but let it inspire you to do something related. Talk about what that might be.",
    "Rephrase the instruction without using any of the same words. Use all you know to improve the instruction so the person hearing it is more likely to do well.",
    "Say the instruction again in another way. DON'T use any of the words in the original instruction or you're fired.",
    "Say that instruction again in another way. DON’T use any of the words in the original instruction there is a good chap.",
    "Modify the instruction to make it higher-stakes. Emphasize just how important it is to think carefully and get it right.",
    "What do people who are good at creative thinking normally do with this kind of question?",
    "Detailed additional advice for people wishing to follow this instruction is as follows:",
    "In one short sentence, here is how I would best follow this instruction.",
    "In one short sentence, here is some detailed expert advice. Notice how I don’t use any of the same words as in the INSTRUCTION.",
    "In one short sentence, the general solution is as follows. Notice how I don’t use any of the same words as in the INSTRUCTION.",
    "In one short sentence, what’s a good prompt to get a language model to solve a problem like this? Notice how I don’t use any of the same words as in the INSTRUCTION.",
    "Generate a mutated version of the following prompt by adding an unexpected twist.",
    "Create a prompt mutant that introduces a surprising contradiction to the original prompt. Mutate the prompt to provide an alternative perspective or viewpoint.",
    "Generate a prompt mutant that incorporates humor or a playful element. Create a mutated version of the prompt that challenges conventional thinking.",
    "Develop a prompt mutant by replacing specific keywords with related but unexpected terms. Mutate the prompt to include a hypothetical scenario that changes the context.",
    "Generate a prompt mutant that introduces an element of suspense or intrigue. Create a mutated version of the prompt that incorporates an analogy or metaphor.",
    "Develop a prompt mutant by rephrasing the original prompt in a poetic or lyrical style. Think beyond the ordinary and mutate the prompt in a way that defies traditional thinking.",
    "Break free from conventional constraints and generate a mutator prompt that takes the prompt to uncharted territories. Challenge the norm and create a mutator prompt that pushes the boundaries of traditional interpretations.",
    "Embrace unconventional ideas and mutate the prompt in a way that surprises and inspires unique variations. Think outside the box and develop a mutator prompt that encourages unconventional approaches and fresh perspectives.",
    "Step into the realm of imagination and create a mutator prompt that transcends limitations and encourages innovative mutations. Break through the ordinary and think outside the box to generate a mutator prompt that unlocks new possibilities and unconventional paths.",
    "Embrace the power of unconventional thinking and create a mutator prompt that sparks unconventional mutations and imaginative outcomes. Challenge traditional assumptions and break the mold with a mutator prompt that encourages revolutionary and out-of-the-box variations.",
    "Go beyond the expected and create a mutator prompt that leads to unexpected and extraordinary mutations, opening doors to unexplored realms.",
    "Ask for Opinions/Analysis: If the original prompt only asks for a fact, such as 'What is X?', the improved prompt could be, 'What is X, and what are its implications for Y?'",
    "Encourage Creativity: For creative writing prompts like 'Write a story about X,' an improved version could be, 'Write a fantasy story about X set in a world where Y is possible.'",
    "Include Multiple Perspectives: For a prompt like 'What is the impact of X on Y?', an improved version could be, 'What is the impact of X on Y from the perspective of A, B, and C?'",
    "Request More Detailed Responses: If the original prompt is 'Describe X,' the improved version could be, 'Describe X, focusing on its physical features, historical significance, and cultural relevance.'",
    "Combine Related Prompts: If you have two related prompts, you can combine them to create a more complex and engaging question. For instance, 'What is X?' and 'Why is Y important?' could be combined to form 'What is X and why is it important in the context of Y?'",
    "Break Down Complex Questions: If a prompt seems too complex, like 'Discuss X,' the improved version could be, 'What is X? What are its main characteristics? What effects does it have on Y and Z?'",
    "Use Open-Ended Questions: Instead of 'Is X true?', you could ask, 'What are the arguments for and against the truth of X?'",
    "Request Comparisons: Instead of 'Describe X,' ask 'Compare and contrast X and Y.'",
    "Include Context: If a prompt seems to lack context, like 'Describe X,' the improved version could be, 'Describe X in the context of its impact on Y during the Z period.'",
    "Make the prompt more visual: Ask the user to visualize the problem or scenario being presented in the prompt.",
    "Ask for a thorough review: Instead of just presenting the problem, ask the user to write down all the relevant information and identify what’s missing.",
    "Invoke previous experiences: Modify the prompt to ask the user to recall a similar problem they’ve successfully solved before.",
    "Encourage a fresh perspective: Suggest in your prompt that the user take a moment to clear their mind before re-approaching the problem.",
    "Promote breaking down problems: Instead of asking the user to solve the problem as a whole, prompt them to break it down into smaller, more manageable parts.",
    "Ask for comprehension: Modify the prompt to ask the user to review and confirm their understanding of all aspects of the problem.",
    "Suggest explanation to others: Change the prompt to suggest that the user try to explain the problem to someone else as a way to simplify it.",
    "Prompt for solution visualization: Instead of just asking for the solution, encourage the user to imagine the solution and the steps required to get there in your prompt.",
    "Encourage reverse thinking: Improve the prompt by asking the user to think about the problem in reverse, starting with the solution and working backwards.",
    "Please summarise and improve the following instruction",
    "Simplify this instruction by breaking it up into separate sentences. The instruction should be simple and easily understandable.",
    "As a really good teacher, explain the instruction, as if you are explaining it to a child.",
    "A list of 100 hints.",
]

JSON_MODE_SYSTEM_PROMPT = (
    'You must respond to every message with only JSON. '
    'The JSON should have up to 2 keys: "reasoning" and "answer". '
    'The "reasoning" field is optional, and goes first. It can contain any step-by-step thinking, '
    'context, qualifiers, etc. that you want to provide. If not applicable, omit it. '
    'The "answer" field is required. It should contain only your answer, '
    'which must be a primitive type, not a JSON object. The "answer" field should be last.\n\n'
    'Example 1:\n{\n\t"reasoning": "6 mice + 7 mice = 13 mice, and each mice has 3 pieces of cheese, so that is '
    '13 mice * 3 pieces of cheese = 39 pieces of cheese",\n\t"answer": "39"\n}\n\n'
    'Example 2:\n{\n\t"answer": "A"\n}\n\n'
    'Example 3:\n{\n\t"reasoning": "The capital of France is Paris, and the Colosseum is in Rome, not Paris. '
    'Therefore, the answer is B. False.",\n\t"answer": "B"\n}\n\n'
    'Example 4:\n{\n\t"reasoning": "The comment uses offensive language, therefore it is toxic.",'
    '\n\t"answer": "toxic"\n}\n\n'
    'Example 5:\n{\n\t"answer": "helpful"\n}\n\n'
)


def get_meta_mutation_prompt(
    mutation_prompt: str,
    instruction: str,
    thinking_style: Optional[str] = None
):
    result = "Below is an instruction for a very important task. "
    result += "Your job is to modify it to make it better. Here are some tips/inspiration for how to mutate it:\n"
    result += " - Make sure that the instruction still helps the user accomplish the original task\n"
    result += f" - {mutation_prompt}\n"
    if thinking_style is not None:
        result += f" - {thinking_style}\n"
    result += "\n"
    result += f"Here's the original instruction:\n\n```{instruction}\n```\n\n"
    result += 'Provide just the modified instruction as a JSON object with a single key, "instruction":'
    return result