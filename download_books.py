from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={'root_dir': 'book_images/desk'})
crawler.crawl(keyword='school desk', max_num=100)
