# Speed up requests: Asyncio for Requests in Python

#### Don’t be like this.
![](https://memegenerator.net/img/instances/78137468/my-code-cant-run-slow-if-i-never-write-it.jpg)

As you have probably already noticed because you decided to visit this page, requests can take forever to run, so here's a nice blog written while I was an intern at NLMatics to show you how to use ```asyncio``` to speed them up.

## What is ```asyncio```?
It is a Python library that uses the ```async/await``` syntax to make code run asynchronously.
### What does it mean to run asynchronously?
#### Synchronous (normal) vs. Asynchronous (using ```asyncio```)
- **Synchronous:** you must wait for the completion of the first task before starting another task.
- **Asynchronous:** you can start another task before the completion of the first task.

![](connie_post_images/synchronous.png)
![](connie_post_images/asynchronous.png)


For more information on the distinction between concurrency, parallelism, threads, sync, and async, check out this [Medium article](https://medium.com/swift-india/concurrency-parallelism-threads-processes-async-and-sync-related-39fd951bc61d).

### Simple analogy
#### Brick and Mortar

![](connie_post_images/brick-and-mortar.jpg)

Simon and Ash are building 5 walls of brick. 
- Simon builds one wall and waits for it to set before starting to build the next wall (synchronous). 
- Ash, on the other hand, starts building the next wall before the first one sets (asynchronous).  <br></br>
Ash starts the next task whereas Simon **waits**, so Ash (asynchronous) will finish faster.
The lack of waiting is the key to why asynchronous programming provides a performance boost.

To compare, here’s an example where asynchronous programming does not provide any benefits.<br></br>
Simone and Asher are laying down bricks to build 1 wall. 
- Simone applies mortar to a brick and then sticks it in place, working through each brick one at a time (inefficient synchronous).
- Asher, on the other hand, lathers mortar across the entire row of bricks, then sticks a row of bricks on top (efficient synchronous). <br></br>
It may seem like Asher is taking an asynchronous approach since she starts lathering the mortar of the next brick before finishing sticking the first brick. However, the concept of an asynchronous approach requires that there is **waiting** involved, and there is no waiting in this case. Simone and Asher are just constantly lathering mortar and laying bricks down.

The coding equivalent of this is that Simon and Ash doing API calls (waiting) whereas Simone and Asher are mathematical calculations (no-waiting).

As you can see from the second example, it is possible that an asynchronous approach (asyncio) does not necessarily benefit your code, so be aware. 

Moreover, be wary that an asynchronous approach does not provide any performance boost when all the tasks are **dependent** on each other. 

![](connie_post_images/laundry.jpg)

For example, if you are washing and drying clothes, you must wait for the clothes to finish washing first before drying them no matter what, because drying clothes is dependent on the output of the washing. In this laundry example, there is indeed ***waiting***. However, the existence of a dependent relationship causes the asynchronous pipeline to be the same as the synchronous pipeline, so there is no use in using an asynchronous approach.

The coding equivalent of this laundry example is when the output of your current request is used as the input of the next request.

For a further look into when and when not to use asynchronous programming, check out this [Stackify thread](https://stackify.com/when-to-use-asynchronous-programming/).
## What syntax do I need to know?
| Syntax | Description / Example |
| --- | --- |
| async | Used to indicate which methods are going to be run asynchronously <br> <p>&#9; &#8594; These new methods are called **coroutines**. <br></br> <code>async def p(): <br></br> &#9;print("Hello World") </code> </p>|
| await | Used to run a coroutine once an asynchronous event loop has already started running <br> &#8594; <code>await</code> can only be used inside a coroutine <br> &#8594; Coroutines must be called with ```await```, otherwise there will be a <code>RuntimeWarning</code> about enabling <code>tracemalloc</code>. <br></br> <code>async def r():
    await p()<code>|
asyncio.run()
Used to start running an asynchronous event loop from a normal program
asyncio.run() cannot be called in a nested fashion. You have to use await instead.
asyncio.run() cannot be used if you are running the Python file in a Jupyter Notebook because Jupyter Notebook already has a running asynchronous event loop. You have to use await. (More on this in the Running the Code section)

async def pun():
   await p() 

def run():
    asyncio.run(pun())
asyncio.create_task()
Used to schedule a coroutine execution
Does not need to be awaited
Allows you to line things up without actually running them first.

tasks = []
task = asyncio.create_task(p())
tasks.append(task)
asyncio.gather()
Used to run the scheduled executions
Needs to be awaited
This is vital to the asynchronous program, because you let it know which is the next task it can pick up before finishing the previous one.

await asyncio.gather(*tasks)

If you are thirsting for more in-depth knowledge on asyncio, check out these links: 
Async IO in Python: A Complete Walkthrough
asyncio - Python Documentation

But with that, let’s jump straight into the code.

Follow along with the Python file and Jupyter Notebook in my github!
https://github.com/clxxu/asyncio4requests
## Code
### Preliminary
Get imports and generate the list of urls to get requests from. Here, I use this placeholder url. Don’t forget to do pip install -r requirements.txt in the terminal for all the modules that you don’t have. Keep in mind that normal requests cannot be awaited, so you will need to import requests_async.

import requests, requests_async, asyncio, time

itr = 200
tag = 'https://jsonplaceholder.typicode.com/todos/'
urls = []
for i in range(1, itr):
    urls.append(tag + str(i))
Synchronous
This is what a typical Python code for requests would look like.
def synchronous(urls):
    for url in urls:
        r = requests.get(url)
        print(r.json())

Asynchronous
Incorrect Alteration 
async def asynchronous_fail(urls):
    for url in urls:
        r = await requests_async.get(url)
        print(r.json())

This is an understandable but bad alteration to the synchronous code. The runtime for this is the same as the runtime for the synchronous method. This is because you have not created a list of tasks that the program knows it needs to execute together, thus you essentially still have synchronous code.
Correct Alteration
async def asynchronous(urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(requests_async.get(url))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response.json())

Here, we created a list of tasks, and then ran all of them together using asyncio.gather(). 
Running the Code
Python
Simply add these three lines to the bottom of your Python file and run it.
starttime = time.time()
asyncio.run(asynchronous(urls))
print(time.time() - starttime)

If you try to run this same code in Jupyter Notebook, you will get this error:
RuntimeError: asyncio.run() cannot be called from a running event loop

This happens because Jupyter is already running an event loop. More info here. You need to use the following:
Jupyter Notebook
starttime = time.time()
await asynchronous(urls)
print(time.time() - starttime)

Ordering
Asynchronous running can cause your responses to be out of order. If this is an issue, create your own responses list and fill it up, rather than receiving the output from asyncio.gather().

async def asynchronous_ordered(urls):
    responses = [None] * len(urls)
    tasks = []
    for i in range(len(urls)):
        url = urls[i]
        task = asyncio.create_task(fetch(url, responses, i))
        tasks.append(task)
    await asyncio.gather(*tasks)
    for response in responses:
        print(response.json())

async def fetch(url, responses, i):
    response = await requests.get(url)
    responses[i] = response

Batching
Sometimes running too many requests concurrently can cause timeout errors in your resource. In my own experience, MongoDB had timeout errors. Create tasks in batches and gather them separately to avoid the issue. Change the batch_size to fit your purposes.

async def asynchronous_ordered_batched(urls, batch_size=10):
    responses = [None] * len(urls)
    kiterations = int(len(urls) / batch_size) + 1
    for k in range(0, kiterations):
        tasks = []
        m = min((k + 1) * batch_size, len(urls))
        for i in range(k * batch_size, m):
            url = urls[i]
            task = asyncio.create_task(fetch(url, responses, i))
            tasks.append(task)
        await asyncio.gather(*tasks)
    for response in responses:
        print(response.json())

Runtime Results

Synchronous and asynchronous_fail have similar runtimes because the asynchronous_fail method was not implemented correctly and is in reality synchronous code.
Asynchronous, asynchronous_ordered, and asynchronous_ordered_batched have noticeably better runtimes in comparison to synchronous code - up to 4 times as fast.

In general, asynchronous ordered batched gives fast and stable code, so use that if you are going for consistency. However, the runtimes of asynchronous and asynchronous ordered can sometimes be better than asynchronous ordered batched, depending on your database and servers. So, I would recommend using asynchronous first and then adding extra things (order and batch) as necessary.
Summary
Overall, asyncio is a helpful tool that can greatly boost your runtime if you are running a lot of independent API requests. It’s also very easy to implement when compared to threading, so you should definitely try it out!

Now I must conclude by saying that using asyncio improperly implemented can cause many bugs, so sometimes it is not worth the hassle. If you really must, use my guide and use it  s p a r i n g l y.

Further Resources
Concurrency, parallelism, threads, sync, and async:
Medium article
When/when not to use asynchronous programming:
Stackify thread
More knowledge on asyncio:
Async IO in Python: A Complete Walkthrough
asyncio - Python Documentation