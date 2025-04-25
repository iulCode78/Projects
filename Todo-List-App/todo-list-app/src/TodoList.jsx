import React, {useState} from "react";

function TodoList(){
    const [tasks, setTasks] = useState(["Make Food"]);
    const [newTask, setNewTasks] = useState("");

    function handleInput(event){
        setNewTasks(event.target.value)
    }
    function handleAddTask(event){
        //const newTask = document.getElementById("taskInput").value;
        //document.getElementById("taskInput").value = "";
        if(newTask.trim() !== ""){
            setTasks(t => [...t, newTask]);
            setNewTasks("");
        }
    }

    function handleDeleteTask(index){
        setTasks(tasks.filter((_, i) => i !== index));
    }

    function updateTaskUp(index){
        if(index > 0){
            const updateTasks = [...tasks];
            [updateTasks[index], updateTasks[index - 1]] = 
            [updateTasks[index - 1], updateTasks[index]];
            setTasks(updateTasks);
        }
    }

    function updateTaskDown(index){
        if(index < tasks.length - 1){
            const updateTasks = [...tasks];
            [updateTasks[index], updateTasks[index + 1]] = 
            [updateTasks[index + 1], updateTasks[index]];
            setTasks(updateTasks);
        }
    }

    return(
        <div className="todo-list-display">
            <div>
                <h1>- Todo List -</h1>
                <input type="text" id="taskInput" placeholder='Add new task' value={newTask} onChange={handleInput}/>
                <button onClick={handleAddTask} id="add-task-btn">+</button>
            </div>
            <ol>
                {tasks.map((task, index) => 
                    <li key={index}><span className="task-text">{task}</span>
                        <button className="delete-task-btn" onClick={() => handleDeleteTask(index)}>-</button>
                        <button className="up-task-btn" onClick={() => updateTaskUp(index)}>Up</button>
                        <button className="down-task-btn" onClick={() => updateTaskDown(index)}>Down</button>
                    </li>
                )}
            </ol>
        </div>
    );
}

export default TodoList