/**
 * --------------------
 * RectangleInsets.java
 * --------------------
 * (C) Copyright 2004, 2005, 2007, by Object Refinery Limited.
 *
 * Original Author:  David Gilbert (for Object Refinery Limited);
 * Contributor(s):   -;
 *
 * $Id: RectangleInsets.java,v 1.15 2007/03/16 14:29:45 mungady Exp $
 *
 * Changes:
 * --------
 * 11-Feb-2004 : Version 1 (DG);
 * 14-Jun-2004 : Implemented Serializable (DG);
 * 02-Feb-2005 : Added new methods and renamed some existing methods (DG);
 * 22-Feb-2005 : Added a new constructor for convenience (DG);
 * 19-Apr-2005 : Changed order of parameters in constructors to match
 *               java.awt.Insets (DG);
 * 16-Mar-2007 : Added default constructor (DG);
 * 21-Jun-2007 : Copied from JCommon (DG);
 * 
 */

package org.jfree.chart.util;

import java.awt.geom.Rectangle2D;
import java.io.Serializable;

/**
 * Represents the insets for a rectangle, specified in absolute or relative 
 * terms. This class is immutable.
 */
public class RectangleInsets implements Serializable {

    /** For serialization. */
    private static final long serialVersionUID = 1902273207559319996L;
    
    /**
     * A useful constant representing zero insets.
     */
    public static final RectangleInsets ZERO_INSETS = new RectangleInsets(
            UnitType.ABSOLUTE, 0.0, 0.0, 0.0, 0.0);
    
    /** Absolute or relative units. */
    private UnitType unitType;
    
    /** The top insets. */
    private double top;
    
    /** The left insets. */
    private double left;
    
    /** The bottom insets. */
    private double bottom;
    
    /** The right insets. */
    private double right;
    
    /**
     * Creates a new instance with all insets initialised to <code>1.0</code>.
     * 
     * @since 1.0.9
     */
    public RectangleInsets() {
        this(1.0, 1.0, 1.0, 1.0);
    }
    
    /**
     * Creates a new instance with the specified insets (as 'absolute' units).
     * 
     * @param top  the top insets.
     * @param left  the left insets.
     * @param bottom  the bottom insets.
     * @param right  the right insets.
     */
    public RectangleInsets(double top, double left, double bottom, 
            double right) {
        this(UnitType.ABSOLUTE, top, left, bottom, right);   
    }
    
    /**
     * Creates a new instance.
     * 
     * @param unitType  absolute or relative units (<code>null</code> not 
     *                  permitted).
     * @param top  the top insets.
     * @param left  the left insets.
     * @param bottom  the bottom insets.
     * @param right  the right insets.
     */
    public RectangleInsets(UnitType unitType, double top, double left, 
                           double bottom, double right) {
        if (unitType == null) {
            throw new IllegalArgumentException("Null 'unitType' argument.");
        }
        this.unitType = unitType;
        this.top = top;
        this.bottom = bottom;
        this.left = left;
        this.right = right;
    }
    
    /**
     * Returns the unit type (absolute or relative).  This specifies whether 
     * the insets are measured as Java2D units or percentages.
     * 
     * @return The unit type (never <code>null</code>).
     */
    public UnitType getUnitType() {
        return this.unitType;
    }
  
    /**
     * Returns the top insets.
     * 
     * @return The top insets.
     */
    public double getTop() {
        return this.top;
    }
    
    /**
     * Returns the bottom insets.
     * 
     * @return The bottom insets.
     */
    public double getBottom() {
        return this.bottom;
    }
    
    /**
     * Returns the left insets.
     * 
     * @return The left insets.
     */
    public double getLeft() {
        return this.left;
    }
    
    /**
     * Returns the right insets.
     * 
     * @return The right insets.
     */
    public double getRight() {
        return this.right;
    }
    
    /**
     * Tests this instance for equality with an arbitrary object.
     * 
     * @param obj  the object (<code>null</code> permitted).
     * 
     * @return A boolean.
     */
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;   
        }
        if (!(obj instanceof RectangleInsets)) {
                return false;
        }
        final RectangleInsets that = (RectangleInsets) obj;
        if (that.unitType != this.unitType) {
            return false;   
        }
        if (this.left != that.left) {
            return false;   
        }
        if (this.right != that.right) {
            return false;   
        }
        if (this.top != that.top) {
            return false;   
        }
        if (this.bottom != that.bottom) {
            return false;   
        }
        return true;   
    }

    /**
     * Returns a hash code for the object.
     * 
     * @return A hash code.
     */
    public int hashCode() {
        int result;
        long temp;
        result = (this.unitType != null ? this.unitType.hashCode() : 0);
        temp = this.top != +0.0d ? Double.doubleToLongBits(this.top) : 0L;
        result = 29 * result + (int) (temp ^ (temp >>> 32));
        temp = this.bottom != +0.0d ? Double.doubleToLongBits(this.bottom) : 0L;
        result = 29 * result + (int) (temp ^ (temp >>> 32));
        temp = this.left != +0.0d ? Double.doubleToLongBits(this.left) : 0L;
        result = 29 * result + (int) (temp ^ (temp >>> 32));
        temp = this.right != +0.0d ? Double.doubleToLongBits(this.right) : 0L;
        result = 29 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    /**
     * Returns a textual representation of this instance, useful for debugging
     * purposes.
     * 
     * @return A string representing this instance.
     */
    public String toString() {
        return "RectangleInsets[t=" + this.top + ",l=" + this.left
                + ",b=" + this.bottom + ",r=" + this.right + "]";
    }
    
    /**
     * Creates an adjusted rectangle using the supplied rectangle, the insets
     * specified by this instance, and the horizontal and vertical 
     * adjustment types.
     * 
     * @param base  the base rectangle (<code>null</code> not permitted).
     * @param horizontal  the horizontal adjustment type (<code>null</code> not
     *                    permitted).
     * @param vertical  the vertical adjustment type (<code>null</code> not 
     *                  permitted).
     * 
     * @return The inset rectangle.
     */
    public Rectangle2D createAdjustedRectangle(Rectangle2D base,
            LengthAdjustmentType horizontal, LengthAdjustmentType vertical) {
        if (base == null) {
            throw new IllegalArgumentException("Null 'base' argument.");
        }
        double x = base.getX();
        double y = base.getY();
        double w = base.getWidth();
        double h = base.getHeight();
        if (horizontal == LengthAdjustmentType.EXPAND) {
            double leftOutset = calculateLeftOutset(w);
            x = x - leftOutset;
            w = w + leftOutset + calculateRightOutset(w);
        }
        else if (horizontal == LengthAdjustmentType.CONTRACT) {
            double leftMargin = calculateLeftInset(w);
            x = x + leftMargin;
            w = w - leftMargin - calculateRightInset(w);
        }
        if (vertical == LengthAdjustmentType.EXPAND) {
            double topMargin = calculateTopOutset(h);
            y = y - topMargin;
            h = h + topMargin + calculateBottomOutset(h);
        }
        else if (vertical == LengthAdjustmentType.CONTRACT) {
            double topMargin = calculateTopInset(h);
            y = y + topMargin;
            h = h - topMargin - calculateBottomInset(h);
        }
        return new Rectangle2D.Double(x, y, w, h);
    }
    
    /**
     * Creates an 'inset' rectangle.
     * 
     * @param base  the base rectangle (<code>null</code> not permitted).
     * 
     * @return The inset rectangle.
     */
    public Rectangle2D createInsetRectangle(Rectangle2D base) {
        return createInsetRectangle(base, true, true);
    }
    
    /**
     * Creates an 'inset' rectangle.
     * 
     * @param base  the base rectangle (<code>null</code> not permitted).
     * @param horizontal  apply horizontal insets?
     * @param vertical  apply vertical insets?
     * 
     * @return The inset rectangle.
     */
    public Rectangle2D createInsetRectangle(Rectangle2D base,
            boolean horizontal, boolean vertical) {
        if (base == null) {
            throw new IllegalArgumentException("Null 'base' argument.");
        }
        double topMargin = 0.0;
        double bottomMargin = 0.0;
        if (vertical) {
            topMargin = calculateTopInset(base.getHeight());
            bottomMargin = calculateBottomInset(base.getHeight());
        }
        double leftMargin = 0.0;
        double rightMargin = 0.0;
        if (horizontal) {
            leftMargin = calculateLeftInset(base.getWidth());
            rightMargin = calculateRightInset(base.getWidth());
        }
        return new Rectangle2D.Double(
            base.getX() + leftMargin, 
            base.getY() + topMargin,
            base.getWidth() - leftMargin - rightMargin,
            base.getHeight() - topMargin - bottomMargin
        );
    }
    
    /**
     * Creates an outset rectangle.
     * 
     * @param base  the base rectangle (<code>null</code> not permitted).
     * 
     * @return An outset rectangle.
     */
    public Rectangle2D createOutsetRectangle(Rectangle2D base) {
        return createOutsetRectangle(base, true, true);
    }
    
    /**
     * Creates an outset rectangle.
     * 
     * @param base  the base rectangle (<code>null</code> not permitted).
     * @param horizontal  apply horizontal insets?
     * @param vertical  apply vertical insets? 
     * 
     * @return An outset rectangle.
     */
    public Rectangle2D createOutsetRectangle(Rectangle2D base,
                                             boolean horizontal, 
                                             boolean vertical) {
        if (base == null) {
            throw new IllegalArgumentException("Null 'base' argument.");
        }
        double topMargin = 0.0;
        double bottomMargin = 0.0;
        if (vertical) {
            topMargin = calculateTopOutset(base.getHeight());
            bottomMargin = calculateBottomOutset(base.getHeight());
        }
        double leftMargin = 0.0;
        double rightMargin = 0.0;
        if (horizontal) {
            leftMargin = calculateLeftOutset(base.getWidth());
            rightMargin = calculateRightOutset(base.getWidth());
        }
        return new Rectangle2D.Double(
            base.getX() - leftMargin, 
            base.getY() - topMargin,
            base.getWidth() + leftMargin + rightMargin,
            base.getHeight() + topMargin + bottomMargin
        );
    }
    
    /**
     * Returns the top margin.
     * 
     * @param height  the height of the base rectangle.
     * 
     * @return The top margin (in Java2D units).
     */
    public double calculateTopInset(double height) {
        double result = this.top;
        if (this.unitType == UnitType.RELATIVE) {
            result = (this.top * height);
        }
        return result;
    }
    
    /**
     * Returns the top margin.
     * 
     * @param height  the height of the base rectangle.
     * 
     * @return The top margin (in Java2D units).
     */
    public double calculateTopOutset(double height) {
        double result = this.top;
        if (this.unitType == UnitType.RELATIVE) {
            result = (height / (1 - this.top - this.bottom)) * this.top;
        }
        return result;
    }
    
    /**
     * Returns the bottom margin.
     * 
     * @param height  the height of the base rectangle.
     * 
     * @return The bottom margin (in Java2D units).
     */
    public double calculateBottomInset(double height) {
        double result = this.bottom;
        if (this.unitType == UnitType.RELATIVE) {
            result = (this.bottom * height);
        }
        return result;
    }

    /**
     * Returns the bottom margin.
     * 
     * @param height  the height of the base rectangle.
     * 
     * @return The bottom margin (in Java2D units).
     */
    public double calculateBottomOutset(double height) {
        double result = this.bottom;
        if (this.unitType == UnitType.RELATIVE) {
            result = (height / (1 - this.top - this.bottom)) * this.bottom;
        }
        return result;
    }

    /**
     * Returns the left margin.
     * 
     * @param width  the width of the base rectangle.
     * 
     * @return The left margin (in Java2D units).
     */
    public double calculateLeftInset(double width) {
        double result = this.left;
        if (this.unitType == UnitType.RELATIVE) {
            result = (this.left * width);
        }
        return result;
    }
    
    /**
     * Returns the left margin.
     * 
     * @param width  the width of the base rectangle.
     * 
     * @return The left margin (in Java2D units).
     */
    public double calculateLeftOutset(double width) {
        double result = this.left;
        if (this.unitType == UnitType.RELATIVE) {
            result = (width / (1 - this.left - this.right)) * this.left;
        }
        return result;
    }
    
    /**
     * Returns the right margin.
     * 
     * @param width  the width of the base rectangle.
     * 
     * @return The right margin (in Java2D units).
     */
    public double calculateRightInset(double width) {
        double result = this.right;
        if (this.unitType == UnitType.RELATIVE) {
            result = (this.right * width);
        }
        return result;
    }
    
    /**
     * Returns the right margin.
     * 
     * @param width  the width of the base rectangle.
     * 
     * @return The right margin (in Java2D units).
     */
    public double calculateRightOutset(double width) {
        double result = this.right;
        if (this.unitType == UnitType.RELATIVE) {
            result = (width / (1 - this.left - this.right)) * this.right;
        }
        return result;
    }
    
    /**
     * Trims the given width to allow for the insets.
     * 
     * @param width  the width.
     * 
     * @return The trimmed width.
     */
    public double trimWidth(double width) {
        return width - calculateLeftInset(width) - calculateRightInset(width);   
    }
    
    /**
     * Extends the given width to allow for the insets.
     * 
     * @param width  the width.
     * 
     * @return The extended width.
     */
    public double extendWidth(double width) {
        return width + calculateLeftOutset(width) + calculateRightOutset(width);   
    }

    /**
     * Trims the given height to allow for the insets.
     * 
     * @param height  the height.
     * 
     * @return The trimmed height.
     */
    public double trimHeight(double height) {
        return height 
               - calculateTopInset(height) - calculateBottomInset(height);   
    }
    
    /**
     * Extends the given height to allow for the insets.
     * 
     * @param height  the height.
     * 
     * @return The extended height.
     */
    public double extendHeight(double height) {
        return height 
               + calculateTopOutset(height) + calculateBottomOutset(height);   
    }

    /**
     * Shrinks the given rectangle by the amount of these insets.
     * 
     * @param area  the area (<code>null</code> not permitted).
     */
    public void trim(Rectangle2D area) {
        double w = area.getWidth();
        double h = area.getHeight();
        double l = calculateLeftInset(w);
        double r = calculateRightInset(w);
    double t = trimHeight(h);
        double b = calculateBottomInset(h);
        area.setRect(area.getX() + l, area.getY() + t, w - l - r, h - t - b);    
    }
    
}
