import sys
from py2neo import Graph
def delete_file_fromdb(filename_to_delete):
    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "123456"
    graph = Graph(url, auth=(username, password))
    try:
        query = """
            MATCH (n {file: $filename_to_delete})
            DETACH DELETE n
            """
        graph.run(query, filename_to_delete=filename_to_delete)
    except Exception as e:
        print("文件删除错误：" + str(e))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python segment_text.py <text>")
        sys.exit(1)
    text = ' '.join(sys.argv[1:])