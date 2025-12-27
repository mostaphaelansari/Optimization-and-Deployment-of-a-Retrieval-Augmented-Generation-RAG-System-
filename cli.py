#!/usr/bin/env python3
"""
RAG System - Command Line Interface
A beautiful CLI for Retrieval Augmented Generation
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_indexer import DocumentIndexer
from vector_store import VectorStore
from document_retriever import DocumentRetriever
from llm_qa_system import LLMQASystem
from evaluator import RAGEvaluator
from chatbot import Chatbot

console = Console()

# ASCII Art Banner
BANNER = """
[bold cyan]
  ╦═╗╔═╗╔═╗  ╔═╗╦ ╦╔═╗╔╦╗╔═╗╔╦╗
  ╠╦╝╠═╣║ ╦  ╚═╗╚╦╝╚═╗ ║ ║╣ ║║║
  ╩╚═╩ ╩╚═╝  ╚═╝ ╩ ╚═╝ ╩ ╚═╝╩ ╩
[/bold cyan]
[dim]Retrieval Augmented Generation System[/dim]
"""


def print_banner():
    """Print the application banner."""
    console.print(BANNER)


def create_status_table(stats: dict) -> Table:
    """Create a status table for vector store stats."""
    table = Table(
        title="📊 Vector Store Status",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        table.add_row(str(key), str(value))
    
    return table


def create_results_table(results: list, title: str = "Search Results") -> Table:
    """Create a formatted results table."""
    table = Table(
        title=f"🔍 {title}",
        show_header=True,
        header_style="bold blue",
        show_lines=True
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Source", style="yellow", width=20)
    table.add_column("Page", style="cyan", width=6)
    table.add_column("Content", style="white", max_width=60)
    
    for r in results:
        content = r['content'][:150].replace('\n', ' ')
        if len(r['content']) > 150:
            content += "..."
        
        table.add_row(
            str(r['rank']),
            f"{r.get('score', 0):.4f}",
            r['metadata'].get('source', 'Unknown'),
            str(r['metadata'].get('page', 'N/A')),
            content
        )
    
    return table


@click.group()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.version_option(version='1.0.0', prog_name='RAG System')
@click.pass_context
def cli(ctx, config):
    """
    🤖 RAG System - Retrieval Augmented Generation CLI
    
    A powerful system for question-answering over PDF documents
    using vector search and local LLMs.
    
    \b
    Quick Start:
      1. Index documents:  python cli.py index data/ -d
      2. Ask questions:    python cli.py ask "Your question here"
      3. Start chatbot:    python cli.py chat
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


# =============================================================================
# INDEX COMMAND
# =============================================================================
@cli.command()
@click.argument('source', type=click.Path(exists=True))
@click.option('--directory', '-d', is_flag=True, help='Source is a directory')
@click.option('--pattern', default='*.pdf', help='Glob pattern for files')
@click.option('--chunk-size', default=None, type=int, help='Override chunk size')
@click.pass_context
def index(ctx, source, directory, pattern, chunk_size):
    """
    📚 Index documents into the vector store.
    
    \b
    Examples:
      python cli.py index data/ -d              # Index all PDFs in data/
      python cli.py index document.pdf          # Index single file
      python cli.py index data/ -d --pattern "*.txt"  # Index text files
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        f"[bold]Indexing documents from:[/] [cyan]{source}[/]",
        title="📚 Document Indexer",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Initialize indexer
        task = progress.add_task("[cyan]Loading embedding model...", total=None)
        indexer = DocumentIndexer(config)
        progress.update(task, description="[green]✓ Embedding model loaded")
        
        # Process documents
        progress.update(task, description="[cyan]Processing documents...")
        chunks = indexer.process_documents(
            source=source,
            is_directory=directory,
            glob_pattern=pattern
        )
        progress.update(task, description=f"[green]✓ Created {len(chunks)} chunks")
        
        # Create vector store
        progress.update(task, description="[cyan]Building vector store...")
        vector_store = VectorStore(
            embeddings=indexer.get_embeddings_model(),
            config_path=config
        )
        vector_store.create_store(chunks)
        progress.update(task, description="[green]✓ Vector store created")
    
    # Show stats
    stats = vector_store.get_collection_stats()
    console.print()
    console.print(create_status_table(stats))
    
    console.print(Panel(
        f"[green]Successfully indexed [bold]{stats['count']}[/] chunks![/]",
        title="✅ Complete",
        border_style="green"
    ))


# =============================================================================
# SEARCH COMMAND
# =============================================================================
@cli.command()
@click.argument('query')
@click.option('--top-k', '-k', default=5, help='Number of results')
@click.option('--threshold', '-t', default=0.0, type=float, help='Minimum score threshold')
@click.pass_context
def search(ctx, query, top_k, threshold):
    """
    🔍 Search documents in the vector store.
    
    \b
    Examples:
      python cli.py search "What is attention?"
      python cli.py search "transformer" -k 10
      python cli.py search "BERT" -t 0.5
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        f"[bold]Query:[/] [cyan]{query}[/]",
        title="🔍 Semantic Search",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Searching...", total=None)
        
        retriever = DocumentRetriever(config_path=config)
        results = retriever.retrieve(query, top_k=top_k, with_scores=True)
        
        # Apply threshold filter
        if threshold > 0:
            results = [r for r in results if r.get('score', 0) >= threshold]
        
        progress.update(task, description=f"[green]✓ Found {len(results)} results")
    
    if results:
        console.print()
        console.print(create_results_table(results))
    else:
        console.print(Panel(
            "[yellow]No results found. Try a different query or lower the threshold.[/]",
            border_style="yellow"
        ))


# =============================================================================
# ASK COMMAND
# =============================================================================
@cli.command()
@click.argument('question')
@click.option('--sources', '-s', is_flag=True, help='Show source documents')
@click.option('--top-k', '-k', default=5, help='Number of context documents')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.pass_context
def ask(ctx, question, sources, top_k, verbose):
    """
    ❓ Ask a question using the RAG system.
    
    \b
    Examples:
      python cli.py ask "What is the Transformer?"
      python cli.py ask "Explain BERT" -s
      python cli.py ask "What is GPT-3?" -s -v
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        f"[bold]Question:[/] [cyan]{question}[/]",
        title="❓ RAG Question-Answering",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=None)
        
        qa_system = LLMQASystem(config)
        
        progress.update(task, description="[cyan]Retrieving documents...")
        progress.update(task, description="[cyan]Generating answer...")
        
        result = qa_system.answer(question, return_sources=sources)
        
        progress.update(task, description="[green]✓ Answer generated")
    
    # Display answer
    console.print()
    console.print(Panel(
        Markdown(result['answer']),
        title="💡 Answer",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Display sources
    if sources and 'sources' in result:
        console.print()
        source_table = Table(
            title="📚 Sources",
            show_header=True,
            header_style="bold yellow"
        )
        source_table.add_column("Document", style="cyan")
        source_table.add_column("Page", style="white")
        source_table.add_column("Score", style="green")
        
        for src in result['sources']:
            source_table.add_row(
                src['source'],
                str(src['page']),
                f"{src['score']:.4f}"
            )
        
        console.print(source_table)
    
    # Verbose output
    if verbose:
        console.print()
        console.print(Panel(
            result['context'][:1000] + "..." if len(result['context']) > 1000 else result['context'],
            title="📄 Context Used",
            border_style="dim"
        ))


# =============================================================================
# CHAT COMMAND
# =============================================================================
@cli.command()
@click.pass_context
def chat(ctx):
    """
    💬 Start interactive chatbot mode.
    
    \b
    Commands in chat:
      /clear  - Clear conversation history
      /sources - Toggle source display
      /help   - Show help
      /quit   - Exit chatbot
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        "[bold]Interactive RAG Chatbot[/]\n\n"
        "Ask questions about your documents.\n"
        "Type [cyan]/help[/] for commands or [cyan]/quit[/] to exit.",
        title="💬 Chatbot",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading chatbot...", total=None)
        chatbot = Chatbot(config)
        progress.update(task, description="[green]✓ Chatbot ready")
    
    console.print()
    show_sources = False
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/]")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().strip()
                
                if cmd == '/quit' or cmd == '/exit':
                    console.print("[dim]Goodbye! 👋[/]")
                    break
                
                elif cmd == '/clear':
                    chatbot.reset_conversation()
                    console.print("[yellow]Conversation cleared.[/]\n")
                    continue
                
                elif cmd == '/sources':
                    show_sources = not show_sources
                    status = "enabled" if show_sources else "disabled"
                    console.print(f"[yellow]Source display {status}.[/]\n")
                    continue
                
                elif cmd == '/help':
                    console.print(Panel(
                        "[cyan]/clear[/]   - Clear conversation history\n"
                        "[cyan]/sources[/] - Toggle source display\n"
                        "[cyan]/help[/]    - Show this help\n"
                        "[cyan]/quit[/]    - Exit chatbot",
                        title="📖 Commands",
                        border_style="blue"
                    ))
                    continue
                
                elif cmd == '/history':
                    history = chatbot.get_conversation_history()
                    if history:
                        for msg in history:
                            role = "You" if msg['role'] == 'user' else "Assistant"
                            console.print(f"[dim]{role}: {msg['content'][:100]}...[/]")
                    else:
                        console.print("[dim]No conversation history.[/]")
                    console.print()
                    continue
                
                else:
                    console.print(f"[red]Unknown command: {cmd}[/]\n")
                    continue
            
            # Generate response
            with console.status("[cyan]Thinking...[/]"):
                result = chatbot.chat(user_input)
            
            # Display response
            console.print(f"\n[bold green]Assistant[/]: {result['response']}\n")
            
            # Show sources if enabled
            if show_sources:
                console.print(f"[dim]Query used: {result['query_used']}[/]")
                console.print(f"[dim]History length: {result['history_length']}[/]\n")
        
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye! 👋[/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]\n")


# =============================================================================
# EVALUATE COMMAND
# =============================================================================
@cli.command()
@click.option('--output', '-o', default='evaluation_results.json', help='Output file')
@click.option('--dataset', '-d', default='data/evaluation_dataset.json', help='Test dataset')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed results')
@click.pass_context
def evaluate(ctx, output, dataset, verbose):
    """
    📊 Evaluate the RAG system performance.
    
    \b
    Examples:
      python cli.py evaluate
      python cli.py evaluate -o results.json -v
      python cli.py evaluate -d custom_test.json
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        f"[bold]Dataset:[/] [cyan]{dataset}[/]\n"
        f"[bold]Output:[/] [cyan]{output}[/]",
        title="📊 RAG Evaluation",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initializing...", total=None)
        
        qa_system = LLMQASystem(config)
        evaluator = RAGEvaluator(qa_system=qa_system, config_path=config)
        
        progress.update(task, description="[cyan]Loading dataset...")
        
        try:
            test_data = evaluator.load_evaluation_dataset(dataset)
            console.print(f"\n[green]Loaded {len(test_data)} test samples[/]\n")
        except FileNotFoundError:
            console.print(f"\n[yellow]Dataset not found, using default samples[/]\n")
            test_data = evaluator.create_evaluation_dataset(
                questions=[
                    "What is the Transformer architecture?",
                    "What does BERT stand for?",
                    "What is few-shot learning?",
                ],
                ground_truths=[
                    "The Transformer is based on self-attention mechanisms.",
                    "BERT stands for Bidirectional Encoder Representations from Transformers.",
                    "Few-shot learning uses a few examples without gradient updates.",
                ],
                expected_sources=[
                    ["1706.03762v7.pdf"],
                    ["1810.04805v2.pdf"],
                    ["2005.14165v4.pdf"],
                ]
            )
    
    # Run evaluation
    console.print("[bold]Running evaluation...[/]\n")
    
    for i, sample in enumerate(test_data, 1):
        with console.status(f"[cyan]Evaluating {i}/{len(test_data)}: {sample.question[:40]}...[/]"):
            pass
        console.print(f"  [green]✓[/] {sample.question[:60]}...")
    
    results = evaluator.evaluate_dataset(test_data, verbose=False)
    
    # Display results
    console.print()
    
    # Retrieval metrics table
    ret_table = Table(title="🔍 Retrieval Metrics", show_header=True, header_style="bold blue")
    ret_table.add_column("Metric", style="cyan")
    ret_table.add_column("Score", style="green")
    ret_table.add_column("Bar", style="white")
    
    for key, value in results.get('retrieval_metrics', {}).items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        ret_table.add_row(key, f"{value:.4f}", bar)
    
    console.print(ret_table)
    console.print()
    
    # Answer metrics table
    ans_table = Table(title="💡 Answer Quality Metrics", show_header=True, header_style="bold green")
    ans_table.add_column("Metric", style="cyan")
    ans_table.add_column("Score", style="green")
    ans_table.add_column("Bar", style="white")
    
    for key, value in results.get('answer_metrics', {}).items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        ans_table.add_row(key, f"{value:.4f}", bar)
    
    console.print(ans_table)
    
    # Save results
    evaluator.save_results(output)
    
    console.print(Panel(
        f"[green]Evaluation complete! Results saved to [bold]{output}[/][/]",
        title="✅ Complete",
        border_style="green"
    ))


# =============================================================================
# STATS COMMAND
# =============================================================================
@cli.command()
@click.pass_context
def stats(ctx):
    """📈 Show vector store statistics."""
    config = ctx.obj['config']
    print_banner()
    
    try:
        indexer = DocumentIndexer(config)
        vector_store = VectorStore(
            embeddings=indexer.get_embeddings_model(),
            config_path=config
        )
        
        stats = vector_store.get_collection_stats()
        console.print(create_status_table(stats))
        
    except FileNotFoundError:
        console.print(Panel(
            "[yellow]Vector store not found.[/]\n\n"
            "Run [cyan]python cli.py index data/ -d[/] first.",
            title="⚠️ Not Indexed",
            border_style="yellow"
        ))


# =============================================================================
# MODELS COMMAND
# =============================================================================
@cli.command()
@click.pass_context
def models(ctx):
    """🤖 List available Ollama models."""
    print_banner()
    
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            
            if models_data:
                table = Table(
                    title="🤖 Available Ollama Models",
                    show_header=True,
                    header_style="bold magenta"
                )
                table.add_column("Model", style="cyan")
                table.add_column("Size", style="green")
                table.add_column("Modified", style="dim")
                
                for m in models_data:
                    size_gb = m.get('size', 0) / (1024**3)
                    table.add_row(
                        m.get('name', 'Unknown'),
                        f"{size_gb:.1f} GB",
                        m.get('modified_at', 'N/A')[:10]
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No models found. Pull a model with:[/]")
                console.print("[cyan]  ollama pull qwen2.5:1.5b[/]")
        else:
            raise Exception("Failed to connect")
            
    except Exception as e:
        console.print(Panel(
            "[red]Cannot connect to Ollama.[/]\n\n"
            "Make sure Ollama is running:\n"
            "[cyan]  ollama serve[/]",
            title="⚠️ Ollama Not Running",
            border_style="red"
        ))


# =============================================================================
# CONFIG COMMAND
# =============================================================================
@cli.command()
@click.pass_context
def config(ctx):
    """⚙️ Show current configuration."""
    config_path = ctx.obj['config']
    print_banner()
    
    from utils import get_config
    cfg = get_config(config_path)
    
    table = Table(
        title="⚙️ Current Configuration",
        show_header=True,
        header_style="bold blue"
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Key settings
    settings = [
        ("LLM Model", cfg.get('llm.model_name', 'N/A')),
        ("Ollama URL", cfg.get('llm.base_url', 'N/A')),
        ("Embedding Model", cfg.get('embeddings.model_name', 'N/A')),
        ("Chunk Size", str(cfg.get('document_processing.chunk_size', 'N/A'))),
        ("Chunk Overlap", str(cfg.get('document_processing.chunk_overlap', 'N/A'))),
        ("Top K", str(cfg.get('retrieval.top_k', 'N/A'))),
        ("Temperature", str(cfg.get('llm.temperature', 'N/A'))),
    ]
    
    for setting, value in settings:
        table.add_row(setting, value)
    
    console.print(table)


# =============================================================================
# WEB COMMAND
# =============================================================================
@cli.command()
@click.option('--port', '-p', default=8501, help='Port number')
@click.pass_context
def web(ctx, port):
    """
    🌐 Launch Streamlit web interface.
    
    \b
    Examples:
      python cli.py web
      python cli.py web -p 8080
    """
    print_banner()
    
    console.print(Panel(
        f"[bold]Starting Streamlit Web Interface[/]\n\n"
        f"Port: [cyan]{port}[/]\n"
        f"URL: [cyan]http://localhost:{port}[/]",
        title="🌐 Web UI",
        border_style="blue"
    ))
    
    import subprocess
    subprocess.run(["streamlit", "run", "app.py", "--server.port", str(port)])



# =============================================================================
# EXPERIMENT COMMAND
# =============================================================================
@cli.command()
@click.option('--output', '-o', default='experiment_results.json', help='Output file for results')
@click.option('--chunk-sizes', '-c', default='256,512,1000', help='Comma-separated chunk sizes to test')
@click.option('--thresholds', '-t', default='0.3,0.4', help='Comma-separated similarity thresholds')
@click.option('--quick', is_flag=True, help='Quick mode with fewer configurations')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.pass_context
def experiment(ctx, output, chunk_sizes, thresholds, quick, verbose):
    """
    🧪 Run RAG experiments with different configurations.
    
    \b
    Examples:
      python cli.py experiment
      python cli.py experiment --quick
      python cli.py experiment -c "512,1000" -t "0.3,0.5" -o results.json
    """
    config = ctx.obj['config']
    print_banner()
    
    # Parse parameters
    chunk_list = [int(x.strip()) for x in chunk_sizes.split(',')]
    threshold_list = [float(x.strip()) for x in thresholds.split(',')]
    
    # Quick mode: use only one embedding model
    if quick:
        embedding_list = ["sentence-transformers/all-MiniLM-L6-v2"]
    else:
        embedding_list = None  # Use all from config
    
    console.print(Panel(
        f"[bold]Chunk Sizes:[/] [cyan]{chunk_list}[/]\n"
        f"[bold]Thresholds:[/] [cyan]{threshold_list}[/]\n"
        f"[bold]Quick Mode:[/] [cyan]{quick}[/]\n"
        f"[bold]Output:[/] [cyan]{output}[/]",
        title="🧪 RAG Experimentation",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading experimentation module...", total=None)
        
        from experimenter import RAGExperimenter
        from evaluator import RAGEvaluator
        
        experimenter = RAGExperimenter(config)
        evaluator = RAGEvaluator(config_path=config)
        
        progress.update(task, description="[cyan]Loading evaluation dataset...")
        
        try:
            test_data = evaluator.load_evaluation_dataset("data/evaluation_dataset.json")
        except FileNotFoundError:
            test_data = evaluator.create_evaluation_dataset(
                questions=[
                    "What is the Transformer architecture?",
                    "What does BERT stand for?",
                    "What is few-shot learning?",
                ],
                ground_truths=[
                    "The Transformer is based on self-attention mechanisms.",
                    "BERT stands for Bidirectional Encoder Representations from Transformers.",
                    "Few-shot learning uses a few examples without gradient updates.",
                ],
                expected_sources=[
                    ["1706.03762v7.pdf"],
                    ["1810.04805v2.pdf"],
                    ["2005.14165v4.pdf"],
                ]
            )
        
        progress.update(task, description=f"[green]✓ Loaded {len(test_data)} test samples")
    
    console.print(f"\n[bold]Running experiments...[/]\n")
    
    # Run experiments
    results = experimenter.run_all_experiments(
        evaluation_samples=test_data,
        chunk_sizes=chunk_list,
        embedding_models=embedding_list,
        thresholds=threshold_list,
        save_intermediate=True
    )
    
    # Generate and save report
    report = experimenter.generate_report(output)
    
    # Display summary
    experimenter.print_summary()
    
    console.print(Panel(
        f"[green]Completed [bold]{len(results)}[/] experiments![/]\n"
        f"Results saved to [cyan]{output}[/]",
        title="✅ Complete",
        border_style="green"
    ))


# =============================================================================
# REWRITE COMMAND
# =============================================================================
@cli.command()
@click.argument('query')
@click.option('--strategy', '-s', default='auto',
              type=click.Choice(['auto', 'expand', 'decompose', 'hyde', 'stepback']))
@click.option('--verbose', '-v', is_flag=True, help='Show rewriting details')
@click.option('--search', is_flag=True, help='Also run search with rewritten query')
@click.pass_context
def rewrite(ctx, query, strategy, verbose, search):
    """
    ✏️ Rewrite a query to improve retrieval.
    
    \b
    Strategies:
      auto      - Automatically select best strategy
      expand    - Add synonyms and related terms
      decompose - Break into simpler sub-queries
      hyde      - Generate hypothetical document
      stepback  - Create broader context query
    
    \b
    Examples:
      python cli.py rewrite "What is BERT?"
      python cli.py rewrite "Compare attention in BERT vs GPT" -s decompose
      python cli.py rewrite "learning rate for training" -s stepback -v
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        f"[bold]Original Query:[/] [cyan]{query}[/]\n"
        f"[bold]Strategy:[/] [cyan]{strategy}[/]",
        title="✏️ Query Rewriter",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading rewriter...", total=None)
        
        from query_rewriter import QueryRewriter
        
        rewriter = QueryRewriter(config_path=config)
        
        progress.update(task, description="[cyan]Rewriting query...")
        
        result = rewriter.rewrite(query, strategy=strategy)
        
        progress.update(task, description="[green]✓ Query rewritten")
    
    console.print()
    
    # Display result
    console.print(Panel(
        f"[bold]Strategy Used:[/] [yellow]{result.strategy_used.upper()}[/]",
        border_style="yellow"
    ))
    
    console.print("\n[bold]Rewritten Queries:[/]\n")
    for i, rq in enumerate(result.rewritten_queries, 1):
        console.print(f"  [cyan]{i}.[/] {rq}")
    
    if verbose and result.metadata:
        console.print(f"\n[dim]Metadata: {result.metadata}[/]")
    
    # Optionally run search with rewritten queries
    if search:
        console.print("\n[bold]Running search with rewritten queries...[/]\n")
        
        retriever = DocumentRetriever(config_path=config)
        
        for rq in result.rewritten_queries[:2]:  # Limit to first 2
            results = retriever.retrieve(rq, top_k=3, with_scores=True)
            if results:
                console.print(f"\n[yellow]Results for:[/] {rq[:50]}...")
                console.print(create_results_table(results, title=f"Query: {rq[:30]}..."))


# =============================================================================
# QUALITY COMMAND (Enhanced Evaluation)
# =============================================================================
@cli.command()
@click.argument('question')
@click.option('--ground-truth', '-g', default=None, help='Expected answer for comparison')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed metrics')
@click.pass_context
def quality(ctx, question, ground_truth, verbose):
    """
    📊 Evaluate answer quality with detailed metrics.
    
    \b
    Metrics:
      - Factuality (grounding in sources)
      - Coherence (logical flow)
      - Precision (accuracy)
    
    \b
    Examples:
      python cli.py quality "What is BERT?"
      python cli.py quality "What is BERT?" -g "Bidirectional Encoder..." -v
    """
    config = ctx.obj['config']
    print_banner()
    
    console.print(Panel(
        f"[bold]Question:[/] [cyan]{question}[/]",
        title="📊 Quality Evaluation",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating answer...", total=None)
        
        from quality_evaluator import QualityEvaluator
        
        qa_system = LLMQASystem(config)
        result = qa_system.answer(question, return_sources=True)
        
        progress.update(task, description="[cyan]Evaluating quality...")
        
        evaluator = QualityEvaluator(config_path=config)
        
        # Use ground truth if provided, otherwise use a placeholder
        gt = ground_truth or "No ground truth provided"
        
        metrics = evaluator.evaluate(
            answer=result['answer'],
            ground_truth=gt,
            context=result['context'],
            question=question
        )
        
        progress.update(task, description="[green]✓ Evaluation complete")
    
    # Display answer
    console.print()
    console.print(Panel(
        Markdown(result['answer']),
        title="💡 Generated Answer",
        border_style="green"
    ))
    
    # Display metrics
    console.print()
    metrics_dict = metrics.to_dict()
    
    for category, values in metrics_dict.items():
        table = Table(
            title=f"📊 {category.upper()} Metrics",
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Bar", style="white")
        
        for metric, score in values.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            table.add_row(metric, f"{score:.4f}", bar)
        
        console.print(table)
        console.print()


if __name__ == '__main__':
    cli()

