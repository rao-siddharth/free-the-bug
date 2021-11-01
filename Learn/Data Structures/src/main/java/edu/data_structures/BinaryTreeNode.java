package main.java.edu.data_structures;

public class BinaryTreeNode {

	private Integer value;
	private BinaryTreeNode left;
	private BinaryTreeNode right;
		
	public Integer getValue() {
		return value;
	}

	public void setValue(Integer value) {
		this.value = value;
	}

	public BinaryTreeNode getLeft() {
		return left;
	}

	public void setLeft(BinaryTreeNode left) {
		this.left = left;
	}

	public BinaryTreeNode getRight() {
		return right;
	}

	public void setRight(BinaryTreeNode right) {
		this.right = right;
	}

	BinaryTreeNode(Integer value){
		this.value = value;
		right = null;
		left = null;
		
	}
	
	@Override
	public String toString() {
		
		StringBuilder sb = new StringBuilder();
		sb.append("Node Value = ").append(this.value);
		sb.append(", Left Child value = ").append(left==null?"Null":left.value);
		sb.append(", Right Child value = ").append(left==null?"Null":right.value);
		
		return sb.toString();
		
	}
}
