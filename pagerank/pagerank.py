import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?\s+)?href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    num_pages = len(corpus)
    links = corpus[page]
    
    if not links:
        # If page has no outgoing links, distribute equally across all pages
        for p in corpus:
            distribution[p] = 1 / num_pages
    else:
        base_prob = (1 - damping_factor) / num_pages
        link_prob = damping_factor / len(links)
        
        for p in corpus:
            distribution[p] = base_prob
            if p in links:
                distribution[p] += link_prob
                
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 0 for page in corpus}
    
    # Choose first sample randomly
    sample = random.choice(list(corpus.keys()))
    ranks[sample] += 1
    
    for i in range(1, n):
        distribution = transition_model(corpus, sample, damping_factor)
        pages = list(distribution.keys())
        weights = list(distribution.values())
        
        # Choose next sample based on probability distribution weights
        sample = random.choices(pages, weights=weights, k=1)[0]
        ranks[sample] += 1
        
    # Convert counts to proportions
    for page in ranks:
        ranks[page] /= n
        
    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    ranks = {page: 1 / num_pages for page in corpus}
    threshold = 0.001
    
    while True:
        new_ranks = {}
        converged = True
        
        for page in corpus:
            rank = (1 - damping_factor) / num_pages
            sigma = 0
            
            for p in corpus:
                corpus_links = corpus[p]
                # A page with no links is interpreted as having links to all pages
                if not corpus_links:
                    sigma += ranks[p] / num_pages
                elif page in corpus_links:
                    sigma += ranks[p] / len(corpus_links)
                    
            rank += damping_factor * sigma
            new_ranks[page] = rank
            
            if abs(new_ranks[page] - ranks[page]) >= threshold:
                converged = False
                
        ranks = new_ranks.copy()
        if converged:
            break
            
    return ranks


if __name__ == "__main__":
    main()