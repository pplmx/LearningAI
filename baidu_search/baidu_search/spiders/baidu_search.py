#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 3/1/2018 16:07
import scrapy
from w3lib.html import remove_tags


class BaiduSearchSpider(scrapy.Spider):
    name = "baidu_search"
    allowed_domains = ["baidu.com"]
    start_urls = ["https://www.baidu.com/s?wd=机器学习"]

    def parse(self, resp):
        href_list = resp.xpath(
            '//div[contains(@class, "c-container")]/h3/a/@href'
        ).extract()
        container_list = resp.xpath('//div[contains(@class, "c-container")]')
        for container in container_list:
            href = container.xpath("h3/a/@href").extract()[0]
            title = remove_tags(container.xpath("h3/a").extract()[0])
            c_abstract = container.xpath(
                'div/div/div[contains(@class, "c-abstract")]'
            ).extract()
            abstract = ""
            if len(c_abstract) > 0:
                abstract = remove_tags(c_abstract[0])
            req = scrapy.Request(href, callback=self.parse_url)
            req.meta["title"] = title
            req.meta["abstract"] = abstract
            yield req

    @staticmethod
    def parse_url(resp):
        print("url:", resp.url)
        print("title:", resp.meta["title"])
        print("abstract:", resp.meta["abstract"])
        content = remove_tags(resp.xpath("//body").extract()[0])
        print("content_length:", len(content))
