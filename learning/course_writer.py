"""Editorial helpers: write only course Markdown under this directory."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = json.loads((HERE / 'rewrite_source_snapshot.json').read_text(encoding='utf8'))['files']


def write(name, text):
    if Path(name).name != name or not name.endswith('.md'):
        raise ValueError('Only direct learning Markdown files may be written')
    (HERE / name).write_text(text.strip() + '\n', encoding='utf8')


def lesson(title, why, intuition, example, steps, definition, mathematics, project, alternatives, question, answer, deeper):
    return f'''## Lesson: {title}

### Why it matters
{why}

### Build the intuition
{intuition}

### A concrete example
{example}

### Walk through it
{steps}

### Give the idea its technical name
{definition}

### Add the mathematics
{mathematics}

### Connect it to our project
{project}

### Alternatives and trade-offs
{alternatives}

### Check your understanding
**Question:** {question}

**Answer explained:** {answer}

### Deeper dive and common confusion
{deeper}
'''


def tables(name):
    """Keep measured/technical tables verbatim while rewriting their teaching context."""
    groups, current = [], []
    for line in OLD[name].splitlines():
        if line.startswith('|'):
            current.append(line)
        elif current:
            groups.append('\n'.join(current)); current = []
    if current:
        groups.append('\n'.join(current))
    return groups


def graph(name):
    import re
    return '\n\n'.join(re.findall(r'```mermaid\n.*?```', OLD[name], re.S))


def sources(name):
    return OLD[name].split('## 11. References and further reading\n')[-1].strip()


def finish(name, intro, lessons, notebook, questions, references=None):
    qa = '\n\n'.join(f'{i}. **{q}**\n\n   {a}' for i,(q,a) in enumerate(questions,1))
    write(name, intro+'\n\n'+'\n\n'.join(lessons)+'\n\n'+notebook+
          '\n\n## Practice explaining it\n\n'+qa+'\n\n## Sources and next lesson\n\n'+(references or sources(name)))
