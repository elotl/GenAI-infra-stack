from common import trim_answer

test_string = """
A recent SSH issue customers had was related to SSH access being flagged as a problem in their old environment because it was under a reseller account. This issue was identified as a "Won\'t Fix" and the Linux Support Engineer informed the customer that they would need to let their security vendor know about these details so they can be excluded from the report as "False Positives."<|im_end|> 

<|im_start|>user
Compose a detailed email to a colleague explaining the importance of using the 'git rebase' command over 'git merge' for integrating changes from a feature branch into the main branch, including the potential impact on the project's history and the benefits of a cleaner commit history.

<|im_end|>second not important<|im_start|> 

## Your task:
Dear [Colleague's Name],
I hope this message finds you well. I wanted to take a moment to discuss an important aspect of our version control practices, specifically regarding the integration of changes from feature branches into our main branch.
As you know, maintaining a clean and understandable project history is crucial for the long-term maintainability and collaboration efficiency of our codebase. This is where the 'git rebase' command comes into play, offering a significant advantage over the traditional 'git merge' approach.
When we use 'git merge', we create a new commit that combines the changes from the feature branch with the main branch. This results in a merge commit, which can clutter our project history with unnecessary merge points. Over time, as we continue to merge feature branches, our history can become a tangled web of commits, making it difficult to track the
"""


def test_clean_all_after_first_im_end():
    assert trim_answer(test_string, "<|im_end|>") == \
        'A recent SSH issue customers had was related to SSH access being flagged as a problem in their old environment because it was under a reseller account. This issue was identified as a "Won\'t Fix" and the Linux Support Engineer informed the customer that they would need to let their security vendor know about these details so they can be excluded from the report as "False Positives."'