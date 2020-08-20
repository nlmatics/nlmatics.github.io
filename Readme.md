# NLMatics blog




## I just wrote a blog post! What next?

Great! 

1. Navigate to the _posts directory.

2. Create a new file called YYYY-MM-DD-NAME-OF-POST.md, replacing YYYY-MM-DD with the date of your post and NAME-OF-POST with the name of your post.

3. Add the following YAML frontmatter to the top of the file, replacing POST TITLE with the post's title, YYYY-MM-DD hh:mm:ss -0000 with the date and time for the post, and CATEGORY-1 and CATEGORY-2 with as many categories you want for your post.

```
layout: page 
title: "POST TITLE" 
date: YYYY-MM-DD hh:mm:ss -0000
categories: CATEGORY-1 CATEGORY-2
```

4. Below the frontmatter, add your post's content. Place your images in the following folder
```
{{site.url}}/site_files/your_post_name/

```



------------------------

We are using Jekyll (https://jekyllrb.com/) to generate a static site from the blog posts.
