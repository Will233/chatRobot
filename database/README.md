## 基于 sqlite 添加问题缓存能力



```python
# mac 中启动 sqlite3
sqlite3
```


创建数据库`chatbot.db`
```bash

sqlite3 ChatBotDB.db
```

创建表 `chats`
```sql
CREATE TABLE chats(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    QUESTION TEXT NOT NULL,
    ANSWER TEXT NOT NULL
);
```

删除表
```python
DROP TABLE chats;
```