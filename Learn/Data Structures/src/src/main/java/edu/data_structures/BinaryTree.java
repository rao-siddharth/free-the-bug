package src.main.java.edu.data_structures;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BinaryTree {

	BinaryTreeNode root;
	
	final int a = 2;
	
	static final String PRINT_TREE_SEPARATOR_SYMBOL = "--|";
	
	public void add(Integer value) {
		root = addNodeToTree(root,value);
	}
	
	private BinaryTreeNode addNodeToTree(BinaryTreeNode node, Integer value) {
		
		if(node == null) {
			return new BinaryTreeNode(value);
		}
		
		if(value <= node.getValue())
			node.setLeft(addNodeToTree(node.getLeft(),value));
		else
			node.setRight(addNodeToTree(node.getRight(),value));
		
		return node;
	}
	
	private void printTree() {
		if(root == null) {
			System.out.println("Empty Tree!");
			return;
		}
		inOrderTraversal(root,0);
	}
	
	private void inOrderTraversal(BinaryTreeNode node,Integer level) {
		if(node == null) {
			return;
		}
		if(node.getLeft()!=null) {
			inOrderTraversal(node.getLeft(),level + 1);
		}
		System.out.print(String.join("",Collections.nCopies(level, PRINT_TREE_SEPARATOR_SYMBOL)));
		System.out.println(node.getValue());

		if(node.getRight()!=null) {
			inOrderTraversal(node.getRight(),level + 1);
		}
	}

	public static void main(String args[]) {
		
		testPrintTree();
		
	}

	private static void testPrintTree() {

		BinaryTree b = new BinaryTree();

		b.add(5);
		b.add(4);
		b.add(6);
		b.add(2);
		b.add(7);
		b.add(8);
		b.add(10);
		b.add(1);
		b.add(3);
		b.add(9);
		b.add(9);
		b.add(16);
		b.add(24);
		b.add(19);

		b.printTree();
	}
	
}
