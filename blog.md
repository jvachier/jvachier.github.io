---
layout: default
title: Blog
---

# Blog

Technical posts about machine learning, deep learning, and research.

---

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url }})
**{{ post.date | date: "%B %d, %Y" }}**

{{ post.excerpt }}

[Read more →]({{ post.url }})

---
{% endfor %}
