from llama_cpp import Llama

# Load the Phi-4-mini-instruct model
llm = Llama(
    model_path="microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",  # Make sure this matches your downloaded file
    n_ctx=4096,  # Phi-4 supports longer context than GPT-2
    n_threads=4,  # Adjust based on your CPU cores
    chat_format="chatml",  # Phi models use ChatML format
    verbose=False
)

def summarize_text(text, max_summary_tokens=200, style="detailed", source=None):
    """
    Summarize text using Phi-4-mini-instruct with optional source attribution
    """
    # Different summary styles
    style_prompts = {
        "detailed": "Please create a comprehensive summary of the following text organized into at least three paragraphs. The first paragraph should introduce the main topic and key findings. The second paragraph should explain the important details and supporting evidence. The third paragraph should discuss the implications, conclusions, or significance of the information.",
        "structured": "Please summarize the following text in exactly three well-developed paragraphs: 1) Main points and key findings, 2) Supporting details and evidence, 3) Implications and conclusions.",
        "comprehensive": "Create a thorough summary of the following text. Organize your response into multiple paragraphs that cover: the main topic and central themes, key details and supporting information, and the broader significance or implications. Each paragraph should be substantive and informative."
    }
    
    # Add source context if provided
    source_instruction = ""
    if source:
        source_instruction = f" When referring to the content, acknowledge that it comes from {source}."
    
    # Create a proper instruction prompt for Phi-4
    messages = [
        {
            "role": "system", 
            "content": f"You are an expert at creating detailed, well-structured summaries. Always organize your summaries into multiple paragraphs with clear, comprehensive content.{source_instruction}"
        },
        {
            "role": "user", 
            "content": f"{style_prompts.get(style, style_prompts['detailed'])}\n\nText to summarize:\n{text}"
        }
    ]
    
    # Generate summary
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_summary_tokens,
        temperature=0.3,  # Low temperature for focused summaries
        top_p=0.8,
        repeat_penalty=1.1
    )
    
    summary = response['choices'][0]['message']['content'].strip()
    
    # Add source attribution at the end if provided
    if source:
        summary += f"\n\n**Source:** {source}"
    
    return summary

def format_markdown_summary(summary, source=None, original_file=None, style=None):
    """Format summary as markdown with proper structure"""
    import datetime
    
    md_content = []
    
    # Add header
    md_content.append("# Summary\n")
    
    # Add metadata
    if original_file:
        md_content.append(f"**Original File:** `{original_file}`  ")
    if source:
        md_content.append(f"**Source:** {source}  ")
    if style:
        md_content.append(f"**Summary Style:** {style.title()}  ")
    
    # Add generation timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content.append(f"**Generated:** {timestamp}  \n")
    
    # Add summary content
    md_content.append("## Summary Content\n")
    md_content.append(summary)
    
    # Add footer
    md_content.append("\n---")
    md_content.append("\n*Summary generated using Phi-4-mini-instruct*")
    
    return "\n".join(md_content)

def save_summary_to_markdown(summary, output_path, source=None, original_file=None, style=None):
    """Save summary to markdown file"""
    try:
        md_content = format_markdown_summary(summary, source, original_file, style)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return True
    except Exception as e:
        print(f"Error saving to markdown: {e}")
        return False
    """Test the summarization with various texts"""
    
    # Test 1: Scientific text
    test_text1 = """
    Scientists have discovered that regular exercise can improve brain function in multiple ways. Exercise increases blood flow to the brain, promotes the growth of new brain cells, and enhances connections between neurons. Studies show that people who exercise regularly have better memory, improved focus, and reduced risk of cognitive decline. The research suggests that even moderate exercise like walking for 30 minutes daily can provide significant benefits for brain health. Additionally, exercise releases chemicals called endorphins that can improve mood and reduce stress levels. Recent longitudinal studies followed participants over 10 years and found that those who maintained regular physical activity showed 40% less cognitive decline compared to sedentary individuals. The mechanisms behind these benefits include increased production of brain-derived neurotrophic factor (BDNF), which supports neuron survival and growth. Furthermore, exercise appears to reduce inflammation in the brain and improve the efficiency of neural networks responsible for executive function and memory formation.
    """
    
    # Test 2: Technology text  
    test_text2 = """
    Artificial intelligence has evolved dramatically over the past decade, transforming from experimental algorithms into practical tools that impact daily life. Machine learning models now power everything from recommendation systems to autonomous vehicles. The development of transformer architectures, particularly models like GPT and BERT, has revolutionized natural language processing. These models can understand context, generate human-like text, and perform complex reasoning tasks. However, challenges remain in areas such as bias mitigation, energy efficiency, and ensuring AI systems remain aligned with human values. The rapid advancement has led to both excitement and concern among researchers and policymakers. Companies are investing billions of dollars in AI research and development, while governments are working to establish regulatory frameworks. The potential applications span healthcare, education, transportation, and scientific research, but ethical considerations around privacy, job displacement, and algorithmic fairness continue to be debated. Looking forward, the integration of AI into society will require careful balance between innovation and responsible deployment.
    """
    
    print("=== Phi-4 Multi-Paragraph Summarization Tests ===\n")
    
    # Test 1 - Detailed style
    print("TEST 1 - Exercise and Brain Function (Detailed Summary):")
    print("Original text:")
    print(test_text1.strip())
    print("\nDetailed Summary:")
    try:
        summary1 = summarize_text(test_text1, max_summary_tokens=250, style="detailed")
        print(summary1)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Test 2 - Structured style
    print("TEST 2 - AI Development (Structured Summary):")
    print("Original text:")
    print(test_text2.strip())
    print("\nStructured Summary:")
    try:
        summary2 = summarize_text(test_text2, max_summary_tokens=250, style="structured")
        print(summary2)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Test 3 - Comprehensive style
    print("TEST 3 - Exercise Text (Comprehensive Summary):")
    print("\nComprehensive Summary:")
    try:
        summary3 = summarize_text(test_text1, max_summary_tokens=300, style="comprehensive")
        print(summary3)
    except Exception as e:
        print(f"Error: {e}")

def interactive_summarization():
    """Interactive mode for custom text input"""
    print("\n=== Interactive Summarization Mode ===")
    print("Choose summary style:")
    print("1. Detailed (3+ paragraphs with comprehensive coverage)")
    print("2. Structured (exactly 3 paragraphs: main points, details, implications)")
    print("3. Comprehensive (multiple well-developed paragraphs)")
    print("\nEnter your text to summarize (or 'quit' to exit):")
    
    while True:
        print("\nSelect style (1-3) or enter text directly:")
        user_input = input("> ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Check if user selected a style
        style_map = {"1": "detailed", "2": "structured", "3": "comprehensive"}
        if user_input in style_map:
            selected_style = style_map[user_input]
            print(f"Selected: {selected_style.title()} style")
            print("Now enter your text:")
            user_text = input("> ")
        else:
            # Treat input as text, use default style
            user_text = user_input
            selected_style = "detailed"
            
        if len(user_text.strip()) < 50:
            print("Please enter a longer text (at least 50 characters) for meaningful summarization")
            continue
            
        try:
            print(f"\nGenerating {selected_style} summary...")
            summary = summarize_text(user_text, max_summary_tokens=300, style=selected_style)
            print(f"\n{selected_style.title()} Summary:")
            print("-" * 40)
            print(summary)
            print("-" * 40)
        except Exception as e:
            print(f"Error generating summary: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_summarization()
        elif sys.argv[1] == "--test":
            test_summarization()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python app.py --test                              # Run test cases")
            print("  python app.py --interactive                       # Interactive mode")
            print("  python app.py --file <path>                       # Summarize a file")
            print("  python app.py --file <path> --style <type>        # Set summary style")
            print("  python app.py --file <path> --source <publication> # Add source attribution")
            print("  python app.py --file <path> --output <file.md>    # Save summary to markdown")
            print()
            print("Styles: detailed (default), structured, comprehensive")
            print()
            print("Examples:")
            print("  python app.py --file article.txt --style comprehensive")
            print("  python app.py --file research.txt --source 'Nature Journal, 2024'")
            print("  python app.py --file report.txt --output summary.md")
            print("  python app.py --file study.txt --style structured --source 'MIT' --output study_summary.md")
        elif sys.argv[1] == "--file" and len(sys.argv) > 2:
            try:
                with open(sys.argv[2], 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Parse optional arguments
                style = "detailed"
                source = None
                output_file = None
                
                # Check for --style argument
                if "--style" in sys.argv:
                    style_index = sys.argv.index("--style")
                    if style_index + 1 < len(sys.argv):
                        style = sys.argv[style_index + 1]
                
                # Check for --source argument
                if "--source" in sys.argv:
                    source_index = sys.argv.index("--source")
                    if source_index + 1 < len(sys.argv):
                        source = sys.argv[source_index + 1]
                
                # Check for --output argument
                if "--output" in sys.argv:
                    output_index = sys.argv.index("--output")
                    if output_index + 1 < len(sys.argv):
                        output_file = sys.argv[output_index + 1]
                
                print(f"Summarizing file: {sys.argv[2]}")
                print(f"Style: {style}")
                if source:
                    print(f"Source: {source}")
                if output_file:
                    print(f"Output: {output_file}")
                print("=" * 60)
                
                # Generate summary
                summary = summarize_text(file_content, max_summary_tokens=300, style=style, source=source)
                
                # Display summary
                print(summary)
                print("=" * 60)
                
                # Save to markdown if output file specified
                if output_file:
                    if save_summary_to_markdown(summary, output_file, source, sys.argv[2], style):
                        print(f"\n✅ Summary saved to: {output_file}")
                    else:
                        print(f"\n❌ Failed to save summary to: {output_file}")
                
            except FileNotFoundError:
                print(f"File not found: {sys.argv[2]}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Unknown argument. Use --help for usage information.")
    else:
        # Default behavior - run tests
        test_summarization()