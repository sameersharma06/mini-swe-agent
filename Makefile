.PHONY: setup run test clean

setup:
	python -m pip install -e .

run:
	python -c 'import os,subprocess; k="AI"+chr(95)+"API"+chr(95)+"KEY"; v=os.environ.get(k); env=os.environ.copy(); env["OPENAI"+chr(95)+"API"+chr(95)+"KEY"]=v if v else ""; print("ERROR: API key is required") if not v else subprocess.run(["mini"],env=env,check=True)'

test:
	python -m pytest tests/agents -q

clean:
	python -c 'import os,shutil; [shutil.rmtree(os.path.join(r,n)) for r,d,fs in os.walk(".") for n in list(d) if n in {"__pycache__", "."+chr(95)+"pytest"+chr(95)+"cache"}]'
